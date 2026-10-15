#!/usr/bin/env python3
"""Export the anchor-strength arms as listening MP3s.

The anchor sweep is the first experiment whose whole point is audible: how much
of the source vocal is fed back in decides how much the *singing* can move while
the lyrics hold.  Numbers can rank the arms, but only listening can say whether
the delivery actually sounds like jazz.

Usage:
    python export_anchor_listening.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av

THESIS_ROOT = Path("/Users/kkszhang/project/毕设")
OUTPUTS = THESIS_ROOT / "experiments" / "nonuniform_noise" / "outputs"
SWEEP = OUTPUTS / "anchor_sweep"
SCORES = OUTPUTS / "anchor_sweep_scores.json"
CLAP = OUTPUTS / "anchor_sweep_clap_text.json"
SOURCE = THESIS_ROOT / "experiments" / "cover_test_kit" / "sources" / "chinese_female_120s.wav"


def load_metrics() -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    if CLAP.is_file():
        for row in json.loads(CLAP.read_text(encoding="utf-8")).get("rows", []):
            vocal = row.get("vocal") or {}
            if row.get("arm") and "jazz" in vocal:
                metrics.setdefault(row["arm"], {})["vocal_jazz"] = vocal["jazz"]
    if SCORES.is_file():
        for row in json.loads(SCORES.read_text(encoding="utf-8")):
            if row.get("arm"):
                metrics.setdefault(row["arm"], {})["coverage"] = row.get("coverage")
    return metrics


def transcode(src: Path, dst: Path) -> None:
    """Re-encode to 44.1 kHz stereo 160 kbps MP3 (libmp3lame via PyAV)."""
    with av.open(str(src)) as inp, av.open(str(dst), "w", format="mp3") as out:
        stream = out.add_stream("libmp3lame", rate=44100)
        stream.bit_rate = 160_000
        stream.layout = "stereo"
        resampler = av.audio.resampler.AudioResampler(
            format="s16p", layout="stereo", rate=44100
        )
        for frame in inp.decode(audio=0):
            for resampled in resampler.resample(frame):
                for packet in stream.encode(resampled):
                    out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUTPUTS / "anchor_listening_mp3")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()
    entries: list[tuple[str, float | None, Path]] = [("00_SOURCE", None, SOURCE)]
    for arm, strength in (("a100", 1.0), ("a075", 0.75), ("a050", 0.5), ("a025", 0.25)):
        wavs = sorted((SWEEP / arm).glob("*.wav"))
        if wavs:
            entries.append((arm, strength, wavs[0]))

    for arm, strength, src in entries:
        if not src.is_file():
            print(f"[skip] {arm}: missing {src}")
            continue
        m = metrics.get(arm, {})
        parts = [arm]
        parts.append("源歌" if strength is None else f"源人声强度{strength:g}")
        if m.get("vocal_jazz") is not None:
            parts.append(f"人声爵士度{m['vocal_jazz']:.3f}")
        if m.get("coverage") is not None:
            parts.append(f"歌词{m['coverage'] * 100:.0f}%")
        dst = args.out / ("_".join(parts) + ".mp3")
        transcode(src, dst)
        print(f"[ok] {dst.name}  ({dst.stat().st_size / 1e6:.1f} MB)")

    print(f"\nwrote previews to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
