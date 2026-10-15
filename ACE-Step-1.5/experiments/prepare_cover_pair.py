"""Align a Covers80 song pair and export equal-duration comparison clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--analysis-sr", type=int, default=16000)
    parser.add_argument("--output-sr", type=int, default=48000)
    return parser.parse_args()


def load_chroma(path: Path, sample_rate: int, hop_length: int):
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    chroma = librosa.feature.chroma_cens(
        y=audio, sr=sample_rate, hop_length=hop_length
    )
    return audio, chroma


def best_alignment(first_chroma, second_chroma, window_frames: int):
    """Return the best local DTW-aligned window, including pitch shift."""
    best = None
    for shift in range(12):
        shifted = np.roll(first_chroma, shift, axis=0)
        _, path = librosa.sequence.dtw(
            X=shifted,
            Y=second_chroma,
            metric="cosine",
            subseq=False,
            backtrack=True,
            global_constraints=True,
            band_rad=0.4,
        )
        path = path[::-1]
        point_cost = 1.0 - np.sum(
            shifted[:, path[:, 0]] * second_chroma[:, path[:, 1]], axis=0
        ) / (
            np.linalg.norm(shifted[:, path[:, 0]], axis=0)
            * np.linalg.norm(second_chroma[:, path[:, 1]], axis=0)
            + 1e-8
        )
        for start in range(0, first_chroma.shape[1] - window_frames + 1):
            stop = start + window_frames
            selected = (path[:, 0] >= start) & (path[:, 0] < stop)
            if selected.sum() < window_frames // 2:
                continue
            mapped = path[selected, 1]
            mapped_span = int(mapped.max() - mapped.min() + 1)
            if not 0.55 * window_frames <= mapped_span <= 1.8 * window_frames:
                continue
            score = float(point_cost[selected].mean())
            candidate = (score, shift, start, int(np.median(mapped)))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("No stable aligned window was found")
    return best


def export_clip(path: Path, center_seconds: float, duration: float, output: Path, sr: int):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    start = max(0, int(round((center_seconds - duration / 2) * sr)))
    stop = start + int(round(duration * sr))
    if stop > len(audio):
        stop = len(audio)
        start = max(0, stop - int(round(duration * sr)))
    clip = audio[start:stop]
    stereo = np.stack([clip, clip], axis=1)
    sf.write(output, stereo, sr, subtype="PCM_16")
    return start / sr


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hop = 4096
    _, first_chroma = load_chroma(args.first, args.analysis_sr, hop)
    _, second_chroma = load_chroma(args.second, args.analysis_sr, hop)
    frames = max(8, int(round(args.duration * args.analysis_sr / hop)))
    score, shift, first_start_frame, second_center_frame = best_alignment(
        first_chroma, second_chroma, frames
    )
    first_center = (first_start_frame + frames / 2) * hop / args.analysis_sr
    second_center = second_center_frame * hop / args.analysis_sr
    first_output = args.output_dir / "source.wav"
    second_output = args.output_dir / "cover.wav"
    first_start = export_clip(
        args.first, first_center, args.duration, first_output, args.output_sr
    )
    second_start = export_clip(
        args.second, second_center, args.duration, second_output, args.output_sr
    )
    manifest = {
        "first": str(args.first),
        "second": str(args.second),
        "source_clip": str(first_output),
        "cover_clip": str(second_output),
        "duration_seconds": args.duration,
        "source_start_seconds": first_start,
        "cover_start_seconds": second_start,
        "pitch_shift_semitones": shift,
        "mean_chroma_distance": score,
    }
    manifest_path = args.output_dir / "alignment.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
