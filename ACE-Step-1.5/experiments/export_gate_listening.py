#!/usr/bin/env python3
"""Convert the B1 context-gate arms into listening MP3s.

The gate is a conditioning-side control, so the thing to listen for is not
"did the lyrics survive" (they do, 96%+ in every arm) but the trade the gate
strikes:

  * accompaniment -- does it actually leave the source style behind?
  * vocal timbre  -- does the singer still sound like the same person?

Each file is named after the arm plus the two numbers a listener is judging, so
the clip can be identified without opening the JSON.

Usage:
    python export_gate_listening.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av

THESIS_ROOT = Path("/Users/kkszhang/project/毕设")
OUTPUTS = THESIS_ROOT / "experiments" / "nonuniform_noise" / "outputs"
GATE_DIR = OUTPUTS / "context_gate"
SCORES = OUTPUTS / "context_gate_scores.json"
CLAP = Path("/tmp/gate_clap.json")

# Arm label -> terms a listener needs in the filename.  Timbre and background
# style distance come from the CLAP pass; coverage from the ASR pass.
ARMS = [
    ("00_SOURCE", None, "源歌（对照）"),
    ("g0_off", 0.0, "门关闭"),
    ("g5_half", 0.5, "门半强度"),
    ("g1_full", 1.0, "门全强度"),
]


def load_metrics() -> dict:
    """Collect clap_bg / clap_timb / coverage per arm, tolerating missing files."""
    metrics: dict[str, dict[str, float]] = {}
    if CLAP.is_file():
        payload = json.loads(CLAP.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload if isinstance(payload, list) else [])
        for row in rows:
            group = row.get("group")
            if group:
                metrics.setdefault(group, {}).update(
                    clap_bg=row.get("clap_background"),
                    clap_timb=row.get("clap_timbre_vocal"),
                )
    if SCORES.is_file():
        for row in json.loads(SCORES.read_text(encoding="utf-8")):
            arm = row.get("arm")
            if arm:
                metrics.setdefault(arm, {})["coverage"] = row.get("coverage")
    return metrics


def wav_for(arm: str) -> Path | None:
    directory = GATE_DIR / arm
    wavs = sorted(directory.glob("*.wav")) if directory.is_dir() else []
    return wavs[0] if wavs else None


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
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUTS / "gate_listening_mp3",
        help="destination directory for the MP3 previews",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()
    source_wav = (
        THESIS_ROOT / "experiments" / "cover_test_kit" / "sources" / "chinese_female_120s.wav"
    )

    for arm, strength, label in ARMS:
        src = source_wav if strength is None else wav_for(arm)
        if src is None or not src.is_file():
            print(f"[skip] {arm}: no wav found")
            continue
        m = metrics.get(arm, {})
        parts = [arm, label]
        if strength is not None:
            parts.append(f"门{strength:g}")
        if m.get("clap_bg") is not None:
            parts.append(f"伴奏改动{m['clap_bg']:.2f}")
        if m.get("clap_timb") is not None:
            parts.append(f"音色保真{m['clap_timb']:.2f}")
        if m.get("coverage") is not None:
            parts.append(f"歌词{m['coverage'] * 100:.0f}%")
        dst = args.out / ("_".join(parts) + ".mp3")
        transcode(src, dst)
        print(f"[ok] {dst.name}  ({dst.stat().st_size / 1e6:.1f} MB)")

    print(f"\nwrote previews to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
