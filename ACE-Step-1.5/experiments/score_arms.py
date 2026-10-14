#!/usr/bin/env python3
"""Score cover arms on source reuse, style change, and lyric coverage.

Lyric coverage alone cannot distinguish "the model sang the lyrics" from "we
wrote the source vocal back into the latent".  The anchor sweep showed the
protected arms reach 93% coverage largely by copying the source (corr_vocal
0.82-0.93), which is a degenerate solution: the accompaniment does change style
(corr_bg stays ~0.1) but the vocals are frozen onto the source, so the edit is
inaudible.

This script therefore reports three numbers per arm:

``corr_vocal``
    Sample correlation with the source over vocal frames.  High means source
    reuse.  The target operating point is a light anchor (~0.2-0.4), and
    anything above ~0.8 is effectively a copy.

``corr_bg``
    Sample correlation with the source over accompaniment frames.  Must stay low
    for the style change to be real.

``coverage``
    ASR lyric coverage (added by ``--with-asr``).

Usage:
    python score_arms.py --dirs sweep/soft_k1 sweep/soft_k3 ...
    python score_arms.py --root outputs/sweep --with-asr
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
THESIS_ROOT = REPO_ROOT.parent
OUTPUTS = THESIS_ROOT / "experiments" / "nonuniform_noise" / "outputs"
LYRICS = THESIS_ROOT / "experiments" / "cover_test_kit" / "03_corrected_lyrics" / "chinese_female.txt"
SOURCE = THESIS_ROOT / "experiments" / "cover_test_kit" / "sources" / "chinese_female_120s.wav"
CJK = re.compile(r"[\u4e00-\u9fff]")


def nmse_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Return normalized MSE in dB between two same-shaped arrays."""
    error = float(np.sum((estimate - reference) ** 2))
    energy = float(np.sum(reference ** 2))
    return 10.0 * np.log10((error + 1e-12) / (energy + 1e-12))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Return the sample correlation of two 1-D arrays."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def band_energy_ratio(x: np.ndarray, sample_rate: int, low: int, high: int) -> float:
    """Return the fraction of spectral energy inside ``[low, high]`` Hz.

    The 6-12 kHz band carries cymbals, hats and other production detail, so its
    level inside *vocal* frames reveals how much of the accompanying music was
    frozen along with the voice -- the temporal mask cannot separate the two.
    """
    n = 2048
    hop = 1024
    if len(x) <= n:
        return float("nan")
    window = np.hanning(n)
    starts = range(0, len(x) - n, hop)
    spectra = np.stack([np.abs(np.fft.rfft(x[i : i + n] * window)) for i in starts])
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    band = (freqs >= low) & (freqs <= high)
    total = float((spectra ** 2).sum())
    if total <= 0.0:
        return float("nan")
    return float((spectra[:, band] ** 2).sum() / total)


def find_wav(directory: Path) -> Optional[Path]:
    """Return the generated wav for one arm directory."""
    manifests = sorted(directory.glob("manifest_*.json"))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for audio in manifest.get("audios", []):
            wav = Path(audio["path"])
            if wav.is_file():
                return wav
    wavs = sorted(directory.glob("*.wav"))
    return wavs[0] if wavs else None


def score_arms(directories: list[Path], reference_audio: np.ndarray, sample_rate: int,
               mask: np.ndarray) -> list[dict]:
    """Compute source-reuse statistics for every arm directory."""
    from evaluate_nonuniform_noise import load_audio

    vocal = mask > 0.5
    background = mask < 0.5
    rows = []
    for directory in directories:
        wav = find_wav(directory)
        if wav is None:
            print(f"  [skip] no wav in {directory}", flush=True)
            continue
        audio, rate = load_audio(wav)
        n = min(len(audio), len(reference_audio))
        audio = audio[:n]
        ref = reference_audio[:n]
        row = {"arm": directory.name, "wav": str(wav)}
        row["corr_vocal"] = correlation(audio[vocal[:n], 0], ref[vocal[:n], 0])
        row["corr_bg"] = correlation(audio[background[:n], 0], ref[background[:n], 0])
        row["nmse_vocal_db"] = nmse_db(audio[vocal[:n]], ref[vocal[:n]])
        row["nmse_bg_db"] = nmse_db(audio[background[:n]], ref[background[:n]])
        row["hf_vocal"] = band_energy_ratio(audio[vocal[:n], 0], rate, 6000, 12000)
        row["hf_bg"] = band_energy_ratio(audio[background[:n], 0], rate, 6000, 12000)
        rows.append(row)
    return rows


def add_asr(rows: list[dict]) -> None:
    """Attach ASR lyric coverage to each row in place."""
    import torch
    from qwen_asr import Qwen3ASRModel

    reference = CJK.findall(LYRICS.read_text(encoding="utf-8"))
    model_dir = THESIS_ROOT / "experiments" / "eval_models" / "Qwen3-ASR-1.7B"
    print(f"loading Qwen3-ASR from {model_dir} ...", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        str(model_dir), dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    )
    for row in rows:
        try:
            out = model.transcribe(audio=row["wav"], language="Chinese", return_time_stamps=False)
            text = out[0].text if isinstance(out, (list, tuple)) else out.text
        except Exception as exc:  # noqa: BLE001
            print(f"  ASR failed for {row['arm']}: {exc}", flush=True)
            continue
        hyp = CJK.findall(text)
        matcher = SequenceMatcher(a=reference, b=hyp, autojunk=False)
        matched_flags = [False] * len(reference)
        for tag, i1, i2, _, _ in matcher.get_opcodes():
            if tag == "equal":
                for k in range(i1, i2):
                    matched_flags[k] = True
        row["coverage"] = sum(matched_flags) / len(reference)
        row["asr_head"] = text[:30]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--root", type=Path, help="score every subdirectory of this root")
    parser.add_argument("--with-asr", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    directories = list(args.dirs)
    if args.root:
        directories += [p for p in sorted(args.root.iterdir()) if p.is_dir()]
    directories = [d for d in directories if d.is_dir()]
    if not directories:
        print("no arm directories given", file=sys.stderr)
        return 1

    from evaluate_nonuniform_noise import load_audio, vocal_sample_mask

    reference_audio, sample_rate = load_audio(SOURCE)
    mask = vocal_sample_mask(len(reference_audio), sample_rate)

    rows = score_arms(directories, reference_audio, sample_rate, mask)
    if args.with_asr:
        add_asr(rows)

    def fmt(value, spec=".3f"):
        if value is None or value != value:
            return "-"
        return format(value, spec)

    header = (
        f"{'arm':<16}{'corr_vocal':>11}{'corr_bg':>9}{'nmse_voc':>10}{'nmse_bg':>9}"
        f"{'hf_vocal':>10}{'hf_bg':>8}"
    )
    if args.with_asr:
        header += f"{'coverage':>10}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        line = (
            f"{row['arm']:<16}{fmt(row['corr_vocal']):>11}{fmt(row['corr_bg']):>9}"
            f"{fmt(row['nmse_vocal_db'], '.2f'):>10}{fmt(row['nmse_bg_db'], '.2f'):>9}"
            f"{fmt(row.get('hf_vocal'), '.4f'):>10}{fmt(row.get('hf_bg'), '.4f'):>8}"
        )
        if "coverage" in row:
            line += f"{row['coverage']*100:>9.1f}%"
        print(line)

    print(
        "\n判读:corr_vocal 越低表示人声被重写越多(源歌自身=1.0, 无保护~0.06);"
        " >0.8 等于复制源人声。corr_bg 应始终低(伴奏真的换了曲风)。"
        " hf_vocal 应脱离源歌水平(源歌约 0.0071):越接近源说明帧内伴奏被一起冻住。"
        " coverage 目标 >=88%。"
    )
    if args.json:
        args.json.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
