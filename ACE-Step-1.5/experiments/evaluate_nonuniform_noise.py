#!/usr/bin/env python3
"""Evaluate cover edits: lyric coverage, style change, and timbre drift.

Three families of metrics, all computed on the *same* 48 kHz stereo outputs:

``coverage``
    ASR the cover, then align its text against the lyrics that were actually
    sung in the source.  This mirrors the archived ``compare_cover_vs_source.py``
    methodology (character tokens for Chinese, word tokens for English) so the
    numbers stay comparable with the earlier ACE-Step / YuE2 tables.

``style``
    Cosine distance between the MERT embedding of the source and of the cover.
    Larger distance means the timbre/texture moved further away from the source,
    i.e. the style change is stronger.  Reported globally and separately over the
    vocal frames and the accompaniment frames, because the whole point of the
    non-uniform schedule is that those two regions should move by different
    amounts.

``timbre``
    Cosine similarity of the frame-wise embedding of the vocal regions only.
    This is the "is it still the same singer" proxy.

Usage:
    python evaluate_nonuniform_noise.py --generate-dir DIR [--generate-dir ...]
    python evaluate_nonuniform_noise.py --asr-only ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
THESIS_ROOT = REPO_ROOT.parent
KIT = THESIS_ROOT / "experiments" / "cover_test_kit"
OUTPUTS = THESIS_ROOT / "experiments" / "nonuniform_noise" / "outputs"
EVAL_MODELS = THESIS_ROOT / "experiments" / "eval_models"
MERT_DIR = Path(os.environ.get("MERT_MODEL_DIR", EVAL_MODELS / "MERT-v1-95M"))
CLAP_DIR = Path(os.environ.get("CLAP_MODEL_DIR", EVAL_MODELS / "clap-htsat-unfused"))
ASR_DIR = Path(os.environ.get("QWEN3_ASR_DIR", EVAL_MODELS / "Qwen3-ASR-1.7B"))

CJK = re.compile(r"[\u4e00-\u9fff]")
FRAME_RATE = 25.0
MERT_RATE = 24000
CLAP_RATE = 48000

SOURCE_WAV = KIT / "sources" / "chinese_female_120s.wav"
ALIGNMENT_JSON = KIT / "02_asr_aligned" / "chinese_female_120s.json"
LYRICS_TXT = KIT / "03_corrected_lyrics" / "chinese_female.txt"


# --------------------------------------------------------------------------- #
# text normalization / coverage
# --------------------------------------------------------------------------- #
def norm_zh(text: str) -> list[str]:
    """Return the Chinese character tokens of ``text``."""
    return CJK.findall(text)


def norm_en(text: str) -> list[str]:
    """Return the lowercased alphanumeric word tokens of ``text``."""
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()


def tokenize(text: str, lang: str) -> list[str]:
    """Tokenize per language using the archived convention."""
    return norm_zh(text) if lang == "zh" else norm_en(text)


def lyric_coverage(reference: str, hypothesis: str, lang: str) -> dict:
    """Align ``hypothesis`` against ``reference`` and report coverage.

    Coverage is the fraction of reference tokens present in the hypothesis, so a
    dropped word lowers it and a repeated word does not raise it.
    """
    ref_tokens = tokenize(reference, lang)
    hyp_tokens = tokenize(hypothesis, lang)
    matcher = SequenceMatcher(a=ref_tokens, b=hyp_tokens, autojunk=False)
    ref_match = [False] * len(ref_tokens)
    hyp_match = [False] * len(hyp_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                ref_match[k] = True
            for k in range(j1, j2):
                hyp_match[k] = True
    matched = sum(ref_match)
    return {
        "reference_tokens": len(ref_tokens),
        "hypothesis_tokens": len(hyp_tokens),
        "matched": matched,
        "dropped": len(ref_tokens) - matched,
        "extra": len(hyp_tokens) - sum(hyp_match),
        "coverage": matched / len(ref_tokens) if ref_tokens else 0.0,
    }


# --------------------------------------------------------------------------- #
# audio loading and the vocal mask
# --------------------------------------------------------------------------- #
def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV as ``(samples, sample_rate)`` with shape ``[time, channels]``."""
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    return audio, sample_rate


def vocal_sample_mask(num_samples: int, sample_rate: int) -> np.ndarray:
    """Rasterize the vocal-activity mask onto the sample grid."""
    alignment = json.loads(ALIGNMENT_JSON.read_text(encoding="utf-8"))
    num_frames = int(np.ceil(num_samples * FRAME_RATE / sample_rate))
    sys.path.insert(0, str(REPO_ROOT))
    from acestep.models.common.vocal_noise_schedule import build_vocal_activity_mask

    frame_mask = build_vocal_activity_mask(alignment, num_frames, fade_seconds=0.25)
    frame_mask = frame_mask.squeeze().numpy()
    samples_per_frame = sample_rate / FRAME_RATE
    idx = np.minimum((np.arange(num_samples) / samples_per_frame).astype(np.int64), num_frames - 1)
    return frame_mask[idx]


def to_mono_24k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Downmix to mono and resample to MERT's 24 kHz input rate."""
    mono = audio.mean(axis=1)
    if sample_rate == MERT_RATE:
        return mono.astype(np.float32)
    import torchaudio

    tensor = torch.from_numpy(mono).float()
    resampled = torchaudio.functional.resample(tensor, sample_rate, MERT_RATE)
    return resampled.numpy().astype(np.float32)


def segment_audio(mono: np.ndarray, *, vocal: bool, sample_rate: int = MERT_RATE) -> np.ndarray:
    """Concatenate the samples belonging to one region, inserting short silences.

    The vocal mask is built for the *duration* of the signal being split, so it
    stays consistent with 48 kHz stereo inputs and 24 kHz mono inputs alike.
    Concatenating (instead of zeroing) keeps the embedding model from seeing long
    silent stretches that would otherwise dominate its pooled statistics.
    """
    # Drop the fade frames so vocal and accompaniment partition the timeline
    # exactly once: vocal is >0.5, accompaniment is <0.5.
    selected_fn = (lambda value: value > 0.5) if vocal else (lambda value: value < 0.5)
    sample_mask = vocal_sample_mask(len(mono), sample_rate)
    samples_per_frame = sample_rate / FRAME_RATE
    n_frames = int(len(mono) // samples_per_frame)
    pieces = []
    for frame in range(n_frames):
        start = int(round(frame * samples_per_frame))
        end = int(round((frame + 1) * samples_per_frame))
        if not selected_fn(float(sample_mask[start])):
            continue
        pieces.append(mono[start:end])
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(0.02 * sample_rate), dtype=np.float32)
    out = [pieces[0]]
    for piece in pieces[1:]:
        out.append(gap)
        out.append(piece)
    return np.concatenate(out).astype(np.float32)


# --------------------------------------------------------------------------- #
# MERT embeddings
# --------------------------------------------------------------------------- #
class MertEmbedder:
    """Thin wrapper over MERT that returns pooled hidden states."""

    def __init__(self, model_dir: Path = MERT_DIR, device: Optional[str] = None) -> None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir))
        self.model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def embed(self, waveform: np.ndarray) -> np.ndarray:
        """Mean-pool MERT's final hidden state over time."""
        if waveform.size == 0:
            raise ValueError("cannot embed an empty waveform")
        inputs = self.extractor(
            waveform, sampling_rate=MERT_RATE, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        hidden = self.model(**inputs, output_hidden_states=True).hidden_states
        pooled = hidden[-1].mean(dim=1).squeeze(0)
        return pooled.float().cpu().numpy()


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return ``1 - cos(a, b)`` for two embedding vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / denom)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return ``cos(a, b)`` for two embedding vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


class ClapEmbedder:
    """CLAP audio embedder, the primary perceptual style-distance metric.

    A music-tagging encoder such as MERT mean-pools away most genre evidence, so
    its headroom between "unchanged source" and "rock re-arrangement" is only a
    few thousandths.  CLAP is contrastively trained against genre/style captions
    and separates those two cases by roughly an order of magnitude more, which is
    what makes it usable for deciding whether the style actually moved.
    """

    def __init__(self, model_dir: Path = CLAP_DIR, device: Optional[str] = None) -> None:
        from transformers import ClapModel, ClapProcessor

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = ClapProcessor.from_pretrained(str(model_dir))
        self.model = ClapModel.from_pretrained(str(model_dir)).eval().to(self.device)

    @torch.no_grad()
    def embed(self, waveform_48k: np.ndarray) -> np.ndarray:
        """Embed a 48 kHz mono waveform and return its pooled audio features."""
        if waveform_48k.size == 0:
            raise ValueError("cannot embed an empty waveform")
        inputs = self.processor(
            audio=waveform_48k, sampling_rate=48000, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return (
            self.model.get_audio_features(**inputs).squeeze(0).float().cpu().numpy()
        )


def to_mono_48k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Downmix to mono at CLAP's expected 48 kHz rate."""
    mono = audio.mean(axis=1)
    if sample_rate == 48000:
        return mono.astype(np.float32)
    import torchaudio

    resampled = torchaudio.functional.resample(
        torch.from_numpy(mono).float(), sample_rate, 48000
    )
    return resampled.numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# ASR
# --------------------------------------------------------------------------- #
class AsrTranscriber:
    """Qwen3-ASR wrapper that transcribes one file at a time."""

    def __init__(self, model_dir: Path = ASR_DIR, device: Optional[str] = None) -> None:
        from qwen_asr import Qwen3ASRModel

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = Qwen3ASRModel.from_pretrained(
            str(model_dir),
            dtype=dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
            max_new_tokens=2048,
        )

    def transcribe(self, wav: Path, language: str = "Chinese") -> str:
        """Transcribe ``wav`` and return the decoded text."""
        result = self.model.transcribe(audio=str(wav), language=language, return_time_stamps=False)
        text = result[0].text if isinstance(result, (list, tuple)) else result.text
        return text.strip()


# --------------------------------------------------------------------------- #
# per-directory evaluation
# --------------------------------------------------------------------------- #
@dataclass
class CoverEval:
    """Metrics for one generated cover."""

    wav: str
    group: str = ""
    coverage: Optional[dict] = None
    style_distance_all: float = float("nan")
    style_distance_vocal: float = float("nan")
    style_distance_background: float = float("nan")
    timbre_similarity_vocal: float = float("nan")
    clap_distance_all: float = float("nan")
    clap_distance_vocal: float = float("nan")
    clap_distance_background: float = float("nan")
    clap_timbre_vocal: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        row = {
            "group": self.group,
            "wav": self.wav,
            "style_all": self.style_distance_all,
            "style_vocal": self.style_distance_vocal,
            "style_background": self.style_distance_background,
            "timbre_vocal": self.timbre_similarity_vocal,
            "clap_all": self.clap_distance_all,
            "clap_vocal": self.clap_distance_vocal,
            "clap_background": self.clap_distance_background,
            "clap_timbre_vocal": self.clap_timbre_vocal,
        }
        if self.coverage:
            row["coverage"] = self.coverage["coverage"]
            row["dropped"] = self.coverage["dropped"]
            row["extra"] = self.coverage["extra"]
            row["ref_tokens"] = self.coverage["reference_tokens"]
        if self.notes:
            row["notes"] = "; ".join(self.notes)
        return row


def find_manifest_wavs(directory: Path) -> list[tuple[str, Path]]:
    """Return ``(group, wav)`` pairs for one arm directory."""
    group = directory.name
    found = []
    for manifest_path in sorted(directory.glob("manifest_*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for audio in manifest.get("audios", []):
            wav = Path(audio["path"])
            if wav.is_file():
                found.append((group, wav))
    if found:
        return found
    return [(group, wav) for wav in sorted(directory.glob("*.wav"))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate-dir",
        type=Path,
        action="append",
        default=[],
        help="directory (or parent of arm directories) holding generated covers",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=None,
        help="only evaluate groups whose name contains one of these substrings",
    )
    parser.add_argument("--style-only", action="store_true", help="skip ASR")
    parser.add_argument("--asr-only", action="store_true", help="skip MERT")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--out", type=Path, default=OUTPUTS / "evaluation.json")
    args = parser.parse_args()

    directories: list[Path] = []
    for raw in args.generate_dir or [OUTPUTS / "ablation"]:
        if (raw / "outputs").is_dir():
            raw = raw / "outputs"
        if any(raw.glob("manifest_*.json")):
            directories.append(raw)
        else:
            directories.extend(
                sorted(p for p in raw.iterdir() if p.is_dir() and any(p.glob("manifest_*.json")))
            )
    if args.patterns:
        directories = [
            d for d in directories if any(p in d.name for p in args.patterns)
        ]
    if not directories:
        print("no generated covers found", file=sys.stderr)
        return 1

    pairs: list[tuple[str, Path]] = []
    for directory in directories:
        pairs.extend(find_manifest_wavs(directory))
    if not pairs:
        print("no wav files found", file=sys.stderr)
        return 1

    source_audio, source_rate = load_audio(SOURCE_WAV)

    embedder = None
    clap = None
    if not args.asr_only:
        print(f"loading MERT from {MERT_DIR} ...", flush=True)
        embedder = MertEmbedder()
        source_mono = to_mono_24k(source_audio, source_rate)
        source_emb = embedder.embed(source_mono)
        source_emb_vocal = embedder.embed(segment_audio(source_mono, vocal=True))
        source_emb_bg = embedder.embed(segment_audio(source_mono, vocal=False))

        print(f"loading CLAP from {CLAP_DIR} ...", flush=True)
        clap = ClapEmbedder()
        source_48k = to_mono_48k(source_audio, source_rate)
        source_clap = clap.embed(source_48k)
        source_clap_vocal = clap.embed(segment_audio(source_48k, vocal=True, sample_rate=CLAP_RATE))
        source_clap_bg = clap.embed(segment_audio(source_48k, vocal=False, sample_rate=CLAP_RATE))

    transcriber = None
    if not args.style_only:
        print(f"loading Qwen3-ASR from {ASR_DIR} ...", flush=True)
        transcriber = AsrTranscriber()
        reference = LYRICS_TXT.read_text(encoding="utf-8")

    rows = []
    for group, wav in pairs:
        entry = CoverEval(wav=str(wav), group=group)
        print(f"[{group}] {wav.name}", flush=True)
        if transcriber is not None:
            try:
                text = transcriber.transcribe(wav, language=args.language)
                entry.coverage = lyric_coverage(reference, text, "zh")
                print(
                    f"    coverage={entry.coverage['coverage']*100:.1f}% "
                    f"dropped={entry.coverage['dropped']} extra={entry.coverage['extra']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - report and keep going
                entry.notes.append(f"asr failed: {type(exc).__name__}: {exc}")
                print(f"    ASR failed: {exc}", flush=True)
        if embedder is not None:
            audio, rate = load_audio(wav)
            if audio.shape != source_audio.shape:
                n = min(len(audio), len(source_audio))
                audio = audio[:n]
                entry.notes.append("truncated to source length")
            mono = to_mono_24k(audio, rate)
            emb = embedder.embed(mono)
            entry.style_distance_all = cosine_distance(source_emb, emb)
            vocal_seg = segment_audio(mono, vocal=True)
            bg_seg = segment_audio(mono, vocal=False)
            if vocal_seg.size:
                emb_v = embedder.embed(vocal_seg)
                entry.style_distance_vocal = cosine_distance(source_emb_vocal, emb_v)
                entry.timbre_similarity_vocal = cosine_similarity(source_emb_vocal, emb_v)
            if bg_seg.size:
                entry.style_distance_background = cosine_distance(
                    source_emb_bg, embedder.embed(bg_seg)
                )
            print(
                f"    style all={entry.style_distance_all:.4f} "
                f"vocal={entry.style_distance_vocal:.4f} "
                f"bg={entry.style_distance_background:.4f} "
                f"timbre_vocal={entry.timbre_similarity_vocal:.4f}",
                flush=True,
            )
        if clap is not None:
            audio, rate = load_audio(wav)
            n = min(len(audio), len(source_audio))
            audio = audio[:n]
            mono48 = to_mono_48k(audio, rate)
            entry.clap_distance_all = cosine_distance(source_clap, clap.embed(mono48))
            vocal_seg = segment_audio(mono48, vocal=True, sample_rate=CLAP_RATE)
            bg_seg = segment_audio(mono48, vocal=False, sample_rate=CLAP_RATE)
            if vocal_seg.size:
                emb_v = clap.embed(vocal_seg)
                entry.clap_distance_vocal = cosine_distance(source_clap_vocal, emb_v)
                entry.clap_timbre_vocal = cosine_similarity(source_clap_vocal, emb_v)
            if bg_seg.size:
                entry.clap_distance_background = cosine_distance(
                    source_clap_bg, clap.embed(bg_seg)
                )
            print(
                f"    clap all={entry.clap_distance_all:.4f} "
                f"vocal={entry.clap_distance_vocal:.4f} "
                f"bg={entry.clap_distance_background:.4f} "
                f"timbre_vocal={entry.clap_timbre_vocal:.4f}",
                flush=True,
            )
        rows.append(entry)

    payload = {
        "source": str(SOURCE_WAV),
        "lyrics": str(LYRICS_TXT),
        "mert_dir": str(MERT_DIR) if embedder is not None else None,
        "clap_dir": str(CLAP_DIR) if clap is not None else None,
        "rows": [r.as_row() for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\nwrote {args.out}")
    print(
        f"{'group':<14}{'coverage':>10}{'clap_all':>10}{'clap_voc':>10}"
        f"{'clap_bg':>10}{'clap_timb':>11}{'mert_all':>10}{'mert_bg':>9}"
    )
    for row in payload["rows"]:
        cov = row.get("coverage")
        cov_s = f"{cov*100:.1f}%" if cov is not None else "-"
        print(
            f"{row['group']:<14}{cov_s:>10}{row['clap_all']:>10.4f}"
            f"{row['clap_vocal']:>10.4f}{row['clap_background']:>10.4f}"
            f"{row['clap_timbre_vocal']:>11.4f}{row['style_all']:>10.4f}"
            f"{row['style_background']:>9.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
