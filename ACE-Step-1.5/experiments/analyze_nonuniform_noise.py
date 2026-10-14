"""Compare global and vocal-aware cover outputs in aligned lyric regions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from acestep.models.common.vocal_noise_schedule import build_vocal_activity_mask


THESIS_ROOT = Path(__file__).resolve().parents[2]
KIT = THESIS_ROOT / "experiments" / "cover_test_kit"
OUTPUTS = THESIS_ROOT / "experiments" / "nonuniform_noise" / "outputs"


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV as float64 stereo samples."""
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    return audio, sample_rate


def weighted_nmse_db(reference: np.ndarray, estimate: np.ndarray, weight: np.ndarray) -> float:
    """Return weighted normalized MSE in decibels; lower means closer."""
    weight_2d = weight[:, None]
    error = np.sum(weight_2d * np.square(estimate - reference))
    energy = np.sum(weight_2d * np.square(reference))
    return float(10.0 * np.log10((error + 1e-12) / (energy + 1e-12)))


def main() -> None:
    """Print source similarity and treatment localization metrics."""
    paths = {
        "global_cns0.60": OUTPUTS / "e95b5c4e-5d70-e933-081c-054aa1d73ec1.wav",
        "global_cns0.25_acs0.20": OUTPUTS
        / "acs0.2_noref"
        / "global"
        / "1df14fda-b04f-c211-1025-41d2171b2d01.wav",
        "adaptive_cns0.25_0.60_acs0.20": OUTPUTS
        / "acs0.2_noref"
        / "adaptive"
        / "1df14fda-b04f-c211-1025-41d2171b2d01.wav",
    }
    source, sample_rate = load_audio(KIT / "sources" / "chinese_female_120s.wav")
    alignment = json.loads(
        (KIT / "02_asr_aligned" / "chinese_female_120s.json").read_text(encoding="utf-8")
    )
    frame_mask = build_vocal_activity_mask(alignment, int(np.ceil(len(source) * 25 / sample_rate)))
    mask = np.repeat(frame_mask.squeeze().numpy(), int(round(sample_rate / 25)))[: len(source)]
    background = 1.0 - mask

    loaded = {}
    for name, path in paths.items():
        audio, output_rate = load_audio(path)
        if output_rate != sample_rate or audio.shape != source.shape:
            raise ValueError(f"Mismatched audio for {name}: {audio.shape}, {output_rate}")
        loaded[name] = audio
        print(
            f"{name}: vocal_nmse={weighted_nmse_db(source, audio, mask):.2f} dB, "
            f"background_nmse={weighted_nmse_db(source, audio, background):.2f} dB"
        )

    treatment = (
        loaded["adaptive_cns0.25_0.60_acs0.20"]
        - loaded["global_cns0.25_acs0.20"]
    )
    vocal_energy = np.sum(mask[:, None] * np.square(treatment)) / (np.sum(mask) + 1e-12)
    background_energy = np.sum(background[:, None] * np.square(treatment)) / (
        np.sum(background) + 1e-12
    )
    print(f"mask_mean={mask.mean():.4f}")
    print(f"adaptive_change_vocal_to_background={vocal_energy / (background_energy + 1e-12):.2f}x")


if __name__ == "__main__":
    main()
