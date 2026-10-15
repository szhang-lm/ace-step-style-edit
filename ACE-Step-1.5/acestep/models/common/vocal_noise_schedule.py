"""Experimental vocal-activity masks and non-uniform cover noise schedules.

The cover sampler in ACE-Step exposes a single global scalar
``cover_noise_strength``.  Everything inside a frame therefore leaves the source
at the same noise level, which couples "how much the style changes" to "how well
the lyrics survive".  This module builds a per-frame vocal-activity mask and
converts it into a *non-uniform* renoising schedule so that vocal frames follow
the source path longer than the accompaniment frames.

Two mechanisms are provided so their contributions can be ablated separately:

``init``
    The vocal frames enter the sampling loop at the vocal noise level instead of
    the global one.  This is the literal "non-uniform initial noise" of the
    method: a single clamped write per frame, at the first step only.

``inject``
    On every step whose global timestep is still above the vocal noise level the
    vocal frames are clamped back onto the source's matched forward-noise path
    ``t * noise + (1 - t) * src``.  Extending this across the loop is the
    dynamic "keep evolving along the source trajectory" variant.

Both mechanisms write exactly the same quantity when they fire; they differ only
in *when*.  ``mode="init"`` applies the write at the first step, ``mode="inject"``
applies it at every later step above the vocal floor, and ``mode="both"`` applies
it at all of them.  Because the write always places the frame on the source path
for the *current* global timestep, the DiT keeps seeing the same in-distribution
per-sample timestep in every mode -- no per-token timestep is ever needed.

Mask quality caveat: aligned lyric timestamps are used as a proxy for vocal
activity.  Token-level timestamps from forced alignment can be badly warped (a
single leading token stretched over many seconds), so ``line`` granularity -- the
union of tokens merged per lyric line -- is the default.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import torch


MASK_PATH_ENV = "ACESTEP_VOCAL_MASK_PATH"
VOCAL_CNS_ENV = "ACESTEP_VOCAL_CNS"
MASK_FADE_ENV = "ACESTEP_VOCAL_MASK_FADE_SEC"
MODE_ENV = "ACESTEP_VOCAL_MODE"
GRANULARITY_ENV = "ACESTEP_VOCAL_GRANULARITY"
INIT_STEPS_ENV = "ACESTEP_VOCAL_INIT_STEPS"
MAX_TOKEN_DURATION_ENV = "ACESTEP_VOCAL_MAX_TOKEN_SEC"
ONSET_FRACTION_ENV = "ACESTEP_VOCAL_ONSET_FRACTION"
ONSET_SECONDS_ENV = "ACESTEP_VOCAL_ONSET_SECONDS"
MASK_NPY_ENV = "ACESTEP_VOCAL_MASK_NPY"
GLOBAL_T_ENV = "ACESTEP_VOCAL_GLOBAL_T"
CONTEXT_STRENGTH_ENV = "ACESTEP_VOCAL_CONTEXT_STRENGTH"
CONTEXT_MASK_NPY_ENV = "ACESTEP_VOCAL_CONTEXT_MASK_NPY"
CONTEXT_TARGET_ENV = "ACESTEP_VOCAL_CONTEXT_TARGET"
ANCHOR_STRENGTH_ENV = "ACESTEP_VOCAL_ANCHOR_STRENGTH"

VALID_MODES = ("none", "init", "inject", "both")
VALID_GRANULARITIES = ("line", "token")
VALID_CONTEXT_TARGETS = ("accompaniment", "vocal")


@dataclass(frozen=True)
class VocalProtectionSpec:
    """Resolved description of how vocal frames are protected during sampling."""

    mask: torch.Tensor
    vocal_timestep: float
    mode: str
    init_steps: int
    granularity: str
    mask_path: Path
    vocal_cns: float
    fade_seconds: float
    # How strongly the source path claims a vocal frame: 1.0 is the hard clamp
    # used by the early arms, 0.0 leaves the frame entirely to the model.  The
    # default keeps every existing caller bit-identical.
    anchor_strength: float = 1.0

    @property
    def protects_init(self) -> bool:
        """Whether the first sampling step writes the source path on vocal frames."""
        return self.mode in ("init", "both")

    def protects_step(self, step_idx: int) -> bool:
        """Whether ``step_idx`` should clamp vocal frames onto the source path."""
        if self.mode == "both":
            return True
        if self.mode == "inject":
            return step_idx > 0
        if self.mode == "init":
            return step_idx < self.init_steps
        return False

    def context_gate(
        self,
        strength: float,
        *,
        mask: Optional[torch.Tensor] = None,
        target: str = "accompaniment",
    ) -> torch.Tensor:
        """Per-frame multiplier for the source block of ``context_latents``.

        The cover branch feeds the source's own latents into the DiT by concatenation
        (``context_latents = cat([src, chunk_masks])``), so a frame-level
        multiplier on that block decides how much source information the
        model is *allowed to read* per frame, as opposed to the noise
        schedule, which decides how far that frame is allowed to drift from
        the source trajectory.

        ``target="accompaniment"`` preserves the existing gate semantics:
        vocal frames retain the source block and accompaniment frames are
        attenuated.  ``target="vocal"`` does the inverse, allowing the vocal
        performance to move while a separate onset mask protects the lyric
        trajectory.  ``mask`` may therefore differ from ``self.mask``.
        """
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"{CONTEXT_STRENGTH_ENV} must be in [0, 1], got {strength}")
        if target not in VALID_CONTEXT_TARGETS:
            raise ValueError(
                f"{CONTEXT_TARGET_ENV} must be one of {VALID_CONTEXT_TARGETS}, got {target!r}"
            )
        context_mask = self.mask if mask is None else mask
        if strength == 0.0:
            return torch.ones_like(context_mask)
        context_mask = context_mask.clamp(0.0, 1.0)
        target_mask = 1.0 - context_mask if target == "accompaniment" else context_mask
        return 1.0 - strength * target_mask


def context_gate_strength_from_env() -> float:
    """Read ``ACESTEP_VOCAL_CONTEXT_STRENGTH``; unset or empty means no gating."""
    raw = os.environ.get(CONTEXT_STRENGTH_ENV, "").strip()
    if not raw:
        return 0.0
    strength = float(raw)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"{CONTEXT_STRENGTH_ENV} must be in [0, 1], got {strength}")
    return strength


def context_gate_target_from_env() -> str:
    """Read which region of the source context should be attenuated."""
    target = os.environ.get(CONTEXT_TARGET_ENV, "accompaniment").strip().lower()
    target = target or "accompaniment"
    if target not in VALID_CONTEXT_TARGETS:
        raise ValueError(
            f"{CONTEXT_TARGET_ENV} must be one of {VALID_CONTEXT_TARGETS}, got {target!r}"
        )
    return target


def _load_precomputed_mask(
    raw_path: str,
    src_latents: torch.Tensor,
    *,
    description: str,
) -> torch.Tensor:
    """Load and batch-expand a one-dimensional per-frame NumPy mask."""
    import numpy as np

    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    values = np.load(path).astype("float32").reshape(-1)
    num_frames = src_latents.shape[1]
    if values.size != num_frames:
        raise ValueError(
            f"{description} has {values.size} frames but the latents have "
            f"{num_frames}; regenerate it for this audio length."
        )
    mask = torch.from_numpy(values).to(
        device=src_latents.device, dtype=src_latents.dtype
    ).view(1, num_frames, 1)
    return mask.expand(src_latents.shape[0], -1, -1).contiguous()


def load_context_gate_mask_from_env(
    src_latents: torch.Tensor,
    fallback_mask: torch.Tensor,
) -> torch.Tensor:
    """Load an optional context-only mask, falling back to the noise mask."""
    raw_path = os.environ.get(CONTEXT_MASK_NPY_ENV, "").strip()
    if not raw_path:
        return fallback_mask
    return _load_precomputed_mask(
        raw_path,
        src_latents,
        description="Precomputed vocal context mask",
    )


def anchor_strength_from_env() -> float:
    """Read ``ACESTEP_VOCAL_ANCHOR_STRENGTH``; unset means the full hard clamp.

    1.0 keeps the historical behaviour (the vocal frame is written onto the
    source path).  Lower values let the frame keep part of its own generated
    value, which is what allows the *performance* to move while the frame is
    still anchored.
    """
    raw = os.environ.get(ANCHOR_STRENGTH_ENV, "").strip()
    if not raw:
        return 1.0
    strength = float(raw)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"{ANCHOR_STRENGTH_ENV} must be in [0, 1], got {strength}")
    return strength


def apply_source_context_gate(
    context_latents: torch.Tensor,
    gate: Optional[torch.Tensor],
    source_channels: int,
) -> torch.Tensor:
    """Scale the source block of ``context_latents`` by a per-frame ``gate``.

    ``context_latents`` is ``cat([src_latents, chunk_masks], dim=-1)`` on the
    channel axis, but the two blocks are *not* equal in width -- the source
    block carries the full latent width (128) while ``chunk_masks`` adds one
    channel per mask.  The boundary therefore has to be passed in by the
    caller, which knows both widths, rather than inferred by halving.

    Only the source block is scaled: ``chunk_masks`` marks region validity,
    and zeroing a validity flag would tell the model that a frame does not
    exist rather than that it carries no source information.
    """
    if gate is None:
        return context_latents
    if not 0 < source_channels < context_latents.shape[-1]:
        raise ValueError(
            "source_channels must split context_latents, got "
            f"{source_channels} of {context_latents.shape[-1]}"
        )
    source = context_latents[..., :source_channels]
    validity = context_latents[..., source_channels:]
    return torch.cat([source * gate.to(source.dtype), validity], dim=-1)


def _clean_interval(start: float, end: float, *, max_duration: float) -> Tuple[float, float]:
    """Return a sanitized interval, repairing zero-width and over-long spans."""
    start = max(0.0, float(start))
    end = max(start, float(end))
    if end - start > max_duration:
        # Forced aligners stretch a whole phrase onto its first token; keep the
        # tail (where the token is actually sung) instead of the full span.
        start = max(0.0, end - max_duration)
    if end == start:
        start = max(0.0, start - 0.04)
        end += 0.04
    return start, end


def _token_interval(token: dict, max_duration: float) -> Optional[Tuple[float, float]]:
    """Return a sanitized token interval, or ``None`` if it is unusable."""
    try:
        start = token["start"]
        end = token["end"]
    except (KeyError, TypeError):
        return None
    if start is None or end is None:
        return None
    try:
        return _clean_interval(float(start), float(end), max_duration=max_duration)
    except (TypeError, ValueError):
        return None


def _line_intervals(
    alignment: dict,
    *,
    max_token_duration: float,
) -> list[Tuple[float, float]]:
    """Return one interval per lyric line, using the line's own bounds.

    Line bounds are the reliable product of the aligner and are what the archived
    verification used (8 lines -> 77.6% of a 120s song protected).  Token-level
    bounds inside a line are deliberately *not* used to trim the span, because a
    forced aligner routinely stretches a phrase onto its first token, which would
    punch holes into genuinely sung passages.
    """
    intervals: list[Tuple[float, float]] = []
    for line in alignment.get("lines") or []:
        if not isinstance(line, dict):
            continue
        try:
            start = float(line["start"])
            end = float(line["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        if end <= start:
            continue
        intervals.append((start, end))

    intervals.sort()
    return intervals


def _token_intervals(alignment: dict, *, max_token_duration: float) -> list[Tuple[float, float]]:
    """Return sanitized token-level intervals."""
    intervals = []
    for token in alignment.get("tokens") or []:
        if not isinstance(token, dict):
            continue
        interval = _token_interval(token, max_token_duration)
        if interval is not None:
            intervals.append(interval)
    intervals.sort()
    return intervals


def extract_intervals(
    alignment: dict,
    *,
    granularity: str = "line",
    max_token_duration: float = 1.5,
) -> list[Tuple[float, float]]:
    """Extract vocal intervals from an aligned-lyrics document."""
    if granularity not in VALID_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {VALID_GRANULARITIES}, got {granularity!r}"
        )
    if granularity == "token":
        return _token_intervals(alignment, max_token_duration=max_token_duration)
    intervals = _line_intervals(alignment, max_token_duration=max_token_duration)
    if intervals:
        return intervals
    # No usable lines -> fall back to sanitized tokens.
    return _token_intervals(alignment, max_token_duration=max_token_duration)


def extract_onset_intervals(
    alignment: dict,
    *,
    granularity: str = "line",
    onset_fraction: Optional[float] = None,
    onset_seconds: Optional[float] = None,
    max_token_duration: float = 1.5,
) -> list[Tuple[float, float]]:
    """Shrink each vocal interval to a short leading window.

    Only the *entry* of a lyric line is anchored.  Word order and "did the line
    happen at all" are decided at the entry, whereas the sustained tail carries
    most of the singing delivery; anchoring just the onset therefore keeps the
    lyrics while leaving the performance free to be re-sung.

    ``onset_fraction`` takes a share of each line, which behaves inconsistently
    across lines of different tempo (a 5s line and a 24s line get very different
    windows).  ``onset_seconds`` takes a fixed duration instead and is the more
    robust knob; if both are given the shorter protection wins.
    """
    if onset_fraction is None and onset_seconds is None:
        raise ValueError("one of onset_fraction or onset_seconds is required")
    if onset_fraction is not None and not 0.0 < onset_fraction <= 1.0:
        raise ValueError(f"onset_fraction must be in (0, 1], got {onset_fraction}")
    if onset_seconds is not None and onset_seconds <= 0.0:
        raise ValueError(f"onset_seconds must be > 0, got {onset_seconds}")

    intervals = extract_intervals(
        alignment, granularity=granularity, max_token_duration=max_token_duration
    )
    shrunk = []
    for start, end in intervals:
        span = end - start
        if span <= 0.0:
            continue
        candidates = [span]
        if onset_fraction is not None:
            candidates.append(span * onset_fraction)
        if onset_seconds is not None:
            candidates.append(float(onset_seconds))
        width = min(candidates)
        if width <= 0.0:
            continue
        shrunk.append((start, start + width))
    return shrunk


def find_energy_peaks(
    curve,
    *,
    max_peaks: int = 3,
    min_gap_frames: int = 2,
    delta: float = 0.02,
) -> list:
    """Return indices of the strongest rises in a 1-D energy curve.

    Works purely in frame units so callers never mix seconds and frames.
    """
    import numpy as np

    energy = np.asarray(curve, dtype=np.float64).reshape(-1)
    if energy.size < 2 or max_peaks < 1:
        return []
    peak_value = float(energy.max())
    if peak_value <= 0.0:
        return []
    threshold = peak_value * delta
    gaps = max(1, int(min_gap_frames))
    peaks: list = []
    idx = 0
    n = energy.size
    while idx < n and len(peaks) < max_peaks:
        if energy[idx] < threshold:
            idx += 1
            continue
        j = idx
        while j + 1 < n and energy[j + 1] >= energy[j]:
            j += 1
        peaks.append(j)
        idx = j + gaps
    return peaks


def extract_latent_onset_windows(
    alignment: dict,
    frame_energy,
    *,
    frames_per_second: float = 25.0,
    granularity: str = "line",
    max_token_duration: float = 1.5,
    max_peaks: int = 3,
    search_seconds: float = 2.0,
    peak_frames: int = 3,
    delta: float = 0.02,
) -> list:
    """Sharpen each lyric interval into windows around its latent onsets.

    Anchoring a fixed duration ("the first 1.0s") spends protection budget on
    whatever falls in that window, including the gaps between syllables.
    Detecting where energy actually rises lets the same budget go further, and
    protection strength trades directly against how much of the delivery can be
    re-sung.

    ``frame_energy`` is a per-frame energy curve at ``frames_per_second`` -- the
    channel magnitude of the source latents works directly, so no audio loading
    or separation model is needed.  Returns ``(start_frame, end_frame)`` pairs.
    """
    import numpy as np

    energy = np.asarray(frame_energy, dtype=np.float64).reshape(-1)
    if energy.size == 0 or max_peaks < 1:
        return []

    intervals = extract_intervals(
        alignment, granularity=granularity, max_token_duration=max_token_duration
    )
    half = max(1, int(peak_frames)) // 2
    search_frames = max(1, int(search_seconds * frames_per_second))
    windows: list = []
    for start_s, end_s in intervals:
        first = int(max(0.0, start_s) * frames_per_second)
        anchor_hi = min(len(energy), first + search_frames)
        last = min(len(energy), int(end_s * frames_per_second))
        if last - first < 2 or anchor_hi - first < 2:
            continue
        peaks = find_energy_peaks(
            energy[first:anchor_hi], max_peaks=max_peaks, min_gap_frames=2, delta=delta
        )
        for offset in peaks:
            frame = first + offset
            windows.append((max(first, frame - half), min(last, frame + half)))
    if not windows:
        return []
    windows.sort()
    merged = [windows[0]]
    for start, end in windows[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def build_vocal_activity_mask(
    alignment: dict,
    num_frames: int,
    *,
    frames_per_second: float = 25.0,
    fade_seconds: float = 0.25,
    max_token_duration: float = 1.5,
    granularity: str = "line",
    onset_fraction: Optional[float] = None,
    onset_seconds: Optional[float] = None,
    frame_energy=None,
    onset_max_peaks: int = 3,
    onset_peak_frames: int = 3,
    onset_search_seconds: float = 2.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a soft ``[1, time, 1]`` vocal-activity mask from aligned lyrics.

    Three shapes are supported, in increasing selectivity:

    ``line`` (no onset option)
        Protect every whole lyric line.  Highest lyric coverage, but the whole
        performance is frozen onto the source.

    ``onset_seconds`` / ``onset_fraction``
        Protect a fixed leading window of each line.  A blunt instrument: the
        budget is spent on whatever falls in the window, gaps included.

    ``frame_energy``
        Protect short windows around the energy onsets detected in the source
        latents.  Same protection budget, aimed at actual syllables.

    The mask is purely temporal: the VAE latent entangles the vocal with the
    accompaniment sharing a frame, so everything inside a protected frame is
    held, including the instruments mixed with the voice.
    """
    import numpy as np

    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive")

    if frame_energy is not None:
        windows = extract_latent_onset_windows(
            alignment,
            frame_energy,
            frames_per_second=frames_per_second,
            granularity=granularity,
            max_token_duration=max_token_duration,
            max_peaks=onset_max_peaks,
            search_seconds=onset_search_seconds,
            peak_frames=onset_peak_frames,
        )
        spans = [
            (start / frames_per_second, end / frames_per_second) for start, end in windows
        ]
    elif onset_fraction is not None or onset_seconds is not None:
        spans = extract_onset_intervals(
            alignment,
            granularity=granularity,
            onset_fraction=onset_fraction,
            onset_seconds=onset_seconds,
            max_token_duration=max_token_duration,
        )
    else:
        spans = extract_intervals(
            alignment, granularity=granularity, max_token_duration=max_token_duration
        )

    frame_times = (
        torch.arange(num_frames, device=device, dtype=torch.float32) + 0.5
    ) / frames_per_second
    mask = torch.zeros(num_frames, device=device, dtype=torch.float32)
    fade_seconds = max(0.0, float(fade_seconds))

    for start, end in spans:
        inside = (frame_times >= start) & (frame_times <= end)
        score = inside.to(torch.float32)
        if fade_seconds > 0:
            distance = torch.maximum(start - frame_times, frame_times - end).clamp_min(0.0)
            edge = distance < fade_seconds
            soft = 0.5 * (1.0 + torch.cos(math.pi * distance / fade_seconds))
            score = torch.where(edge, soft, score)
        mask = torch.maximum(mask, score)

    return mask.to(dtype=dtype).view(1, num_frames, 1)


def nearest_timestep(valid_timesteps: Iterable[float], target: float) -> float:
    """Snap ``target`` onto the nearest valid sampler timestep."""
    candidates = list(valid_timesteps)
    if not candidates:
        raise ValueError("valid_timesteps must not be empty")
    return min(candidates, key=lambda value: abs(value - target))


def resolve_mode(raw: Optional[str]) -> str:
    """Normalize a mode string, treating an empty value as ``none``."""
    if raw is None:
        return "none"
    mode = raw.strip().lower()
    if not mode:
        return "none"
    if mode not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be one of {VALID_MODES}, got {raw!r}")
    return mode


def load_vocal_schedule_from_env(
    src_latents: torch.Tensor,
    *,
    valid_timesteps: Optional[Sequence[float]] = None,
    global_timestep: Optional[float] = None,
) -> Optional[VocalProtectionSpec]:
    """Load an opt-in vocal protection schedule from the environment.

    Returns ``None`` when no mask path is configured, so the sampler stays on its
    stock global-scalar behaviour.
    """
    # Either a precomputed per-frame mask or a lyric alignment may drive the
    # schedule; a precomputed mask wins when both are configured.
    raw_npy = os.environ.get(MASK_NPY_ENV, "").strip()
    raw_path = os.environ.get(MASK_PATH_ENV, "").strip()
    if not raw_npy and not raw_path:
        return None
    path: Optional[Path] = None
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Vocal alignment file not found: {path}")

    mode = resolve_mode(os.environ.get(MODE_ENV))
    vocal_cns = float(os.environ.get(VOCAL_CNS_ENV, "0.6"))
    if not 0.0 <= vocal_cns <= 1.0:
        raise ValueError(f"{VOCAL_CNS_ENV} must be in [0, 1], got {vocal_cns}")
    fade_seconds = float(os.environ.get(MASK_FADE_ENV, "0.25"))
    if fade_seconds < 0.0:
        raise ValueError(f"{MASK_FADE_ENV} must be >= 0, got {fade_seconds}")
    granularity = os.environ.get(GRANULARITY_ENV, "line").strip().lower() or "line"
    if granularity not in VALID_GRANULARITIES:
        raise ValueError(
            f"{GRANULARITY_ENV} must be one of {VALID_GRANULARITIES}, got {granularity!r}"
        )
    max_token_duration = float(os.environ.get(MAX_TOKEN_DURATION_ENV, "1.5"))
    init_steps = int(os.environ.get(INIT_STEPS_ENV, "1"))
    if init_steps < 1:
        raise ValueError(f"{INIT_STEPS_ENV} must be >= 1, got {init_steps}")
    raw_onset = os.environ.get(ONSET_FRACTION_ENV, "").strip()
    onset_fraction: Optional[float] = None
    if raw_onset:
        onset_fraction = float(raw_onset)
        if not 0.0 < onset_fraction <= 1.0:
            raise ValueError(
                f"{ONSET_FRACTION_ENV} must be in (0, 1], got {onset_fraction}"
            )
    raw_onset_sec = os.environ.get(ONSET_SECONDS_ENV, "").strip()
    onset_seconds: Optional[float] = None
    use_latent_onsets = raw_onset_sec.lower() == "energy"
    if raw_onset_sec and not use_latent_onsets:
        onset_seconds = float(raw_onset_sec)
        if onset_seconds <= 0.0:
            raise ValueError(f"{ONSET_SECONDS_ENV} must be > 0, got {onset_seconds}")
    frame_energy = None
    if use_latent_onsets:
        frame_energy = src_latents.detach().float().abs().mean(dim=-1)[0].cpu().numpy()

    if raw_npy:
        mask = _load_precomputed_mask(
            raw_npy,
            src_latents,
            description="Precomputed vocal mask",
        )
    else:
        if path is None:
            raise ValueError(
                f"{MASK_PATH_ENV} is required when {MASK_NPY_ENV} is not set"
            )
        alignment = json.loads(path.read_text(encoding="utf-8"))
        mask = build_vocal_activity_mask(
            alignment,
            src_latents.shape[1],
            fade_seconds=fade_seconds,
            max_token_duration=max_token_duration,
            granularity=granularity,
            onset_fraction=onset_fraction,
            onset_seconds=onset_seconds,
            frame_energy=frame_energy,
            device=src_latents.device,
            dtype=src_latents.dtype,
        )
        mask = mask.expand(src_latents.shape[0], -1, -1)

    # ``vocal_cns`` is a cover-noise strength: larger means closer to the source,
    # i.e. a *lower* noise level.
    requested = 1.0 - vocal_cns
    if valid_timesteps:
        vocal_timestep = nearest_timestep(valid_timesteps, requested)
    else:
        vocal_timestep = requested

    if global_timestep is not None and vocal_timestep > global_timestep + 1e-6:
        raise ValueError(
            "The vocal noise level must not exceed the global cover noise level: "
            f"vocal_t={vocal_timestep:.4f} > global_t={global_timestep:.4f}. "
            "Set ACESTEP_VOCAL_CNS >= cover_noise_strength so vocal frames are the "
            "less-noisy region."
        )

    # Fail on a malformed context-gate strength here rather than mid-sample.
    context_gate_strength_from_env()
    context_gate_target_from_env()

    return VocalProtectionSpec(
        mask=mask,
        vocal_timestep=vocal_timestep,
        mode=mode,
        init_steps=init_steps,
        granularity=granularity,
        mask_path=path,
        vocal_cns=vocal_cns,
        fade_seconds=fade_seconds,
        anchor_strength=anchor_strength_from_env(),
    )

def source_path_at(
    src_latents: torch.Tensor,
    noise: torch.Tensor,
    timestep: float,
) -> torch.Tensor:
    """Return the source's matched forward-noise latent at ``timestep``."""
    return timestep * noise + (1.0 - timestep) * src_latents


def inject_source_path(
    xt: torch.Tensor,
    src_latents: torch.Tensor,
    noise: torch.Tensor,
    vocal_mask: torch.Tensor,
    timestep: float,
    strength: float = 1.0,
) -> torch.Tensor:
    """Blend vocal frames towards the source's matched forward-noise path.

    ``strength`` is a continuous anchor weight rather than a switch:

    * ``1.0`` replaces the vocal frames with the source path (the hard clamp the
      earlier arms used),
    * ``0.0`` leaves ``xt`` untouched,
    * in between the frame keeps ``1 - strength`` of its own generated value and
      takes ``strength`` of the source path.

    The middle of that range is what "reduce how strongly the source vocal is
    fed back in" means concretely: the source supplies an anchor without owning
    the frame, so the performance can still move.
    """
    if strength <= 0.0:
        return xt
    blend = vocal_mask * strength
    return torch.lerp(xt, source_path_at(src_latents, noise, timestep), blend)


def prepare_nonuniform_start(
    src_latents: torch.Tensor,
    noise: torch.Tensor,
    spec: VocalProtectionSpec,
    global_timestep: float,
) -> torch.Tensor:
    """Build the non-uniform starting latent for a cover run.

    Accompaniment frames start at ``global_timestep``; vocal frames start part
    way towards the lower ``spec.vocal_timestep`` as weighted by
    ``spec.anchor_strength``.  With ``spec.mode == "none"`` this reduces exactly
    to ``renoise(src_latents, global_timestep, noise)``.
    """
    xt = source_path_at(src_latents, noise, global_timestep)
    if spec.mode == "none":
        return xt
    return inject_source_path(
        xt,
        src_latents,
        noise,
        spec.mask,
        spec.vocal_timestep,
        strength=spec.anchor_strength,
    )
