#!/usr/bin/env python3
"""Export the vocal-context release arms as descriptive listening MP3s."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av

THESIS_ROOT = Path("/Users/kkszhang/project/毕设")
OUTPUTS = THESIS_ROOT / "experiments" / "nonuniform_noise" / "outputs"
SWEEP = OUTPUTS / "vocal_context_release"
SCORES = OUTPUTS / "vocal_context_release_scores.json"
SOURCE = THESIS_ROOT / "experiments" / "cover_test_kit" / "sources" / "chinese_female_120s.wav"

ARMS = (
    ("vctx100", 1.0),
    ("vctx075", 0.75),
    ("vctx050", 0.5),
    ("vctx000", 0.0),
)


def load_scores() -> dict[str, dict[str, float]]:
    """Index the scalar evaluation results by experiment arm."""
    if not SCORES.is_file():
        return {}
    return {row["arm"]: row for row in json.loads(SCORES.read_text(encoding="utf-8"))}


def transcode(src: Path, dst: Path) -> None:
    """Re-encode a WAV preview to 44.1 kHz stereo 160 kbps MP3."""
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
    """Write the source and four generated arms with metrics in each filename."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUTS / "vocal_context_release_listening_mp3",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scores = load_scores()
    entries: list[tuple[str, float | None, Path]] = [("00_SOURCE", None, SOURCE)]
    for arm, retained in ARMS:
        wavs = sorted((SWEEP / arm).glob("*.wav"))
        if wavs:
            entries.append((arm, retained, wavs[0]))

    for arm, retained, src in entries:
        row = scores.get(arm, {})
        parts = [arm, "源歌" if retained is None else f"源人声条件{retained:g}"]
        if row.get("corr_vocal") is not None:
            parts.append(f"人声锚定{row['corr_vocal']:.3f}")
        if row.get("coverage") is not None:
            parts.append(f"歌词{row['coverage'] * 100:.0f}%")
        dst = args.out / ("_".join(parts) + ".mp3")
        transcode(src, dst)
        print(f"[ok] {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")

    print(f"wrote previews to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
