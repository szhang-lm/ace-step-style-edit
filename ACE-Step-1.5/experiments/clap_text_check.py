#!/usr/bin/env python3
"""Ask CLAP the question the distance metrics cannot: *which style is this?*

The gate experiments originally reported ``clap_bg``, the CLAP distance between
an arm's accompaniment and the source's.  That number only says the
accompaniment moved away from the source; it cannot say whether it moved
*towards the requested style* or simply fell apart.  Listening found the
failure mode the metric hid: at full gate strength the accompaniment became
percussive noise, not jazz.

This script scores each arm against explicit style descriptions and reports the
ranking, which is the property a listener actually judges.  It also scores the
source, because the source is already jazz: an arm that "changed the style"
should move *away* from the source's text similarity, so the source column is
the reference point, not a target.

Usage:
    python clap_text_check.py --dir DIR [--out JSON]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_nonuniform_noise import (  # noqa: E402
    CLAP_DIR,
    OUTPUTS,
    ClapEmbedder,
    load_audio,
    segment_audio,
    to_mono_48k,
)

PROMPTS = {
    "jazz": "smooth jazz with upright piano, brushed drums and double bass",
    "rock": "rock music with distorted electric guitar and heavy drums",
    "percussion": "loud percussion and drum solo, cymbals and tom toms",
    "ballad": "quiet slow ballad, soft piano and strings",
    "electronic": "electronic dance music with synthesizer and drum machine",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUTPUTS / "gate_clap_text.json")
    args = parser.parse_args()

    import torch

    print(f"loading CLAP from {CLAP_DIR} ...", flush=True)
    clap = ClapEmbedder()

    text_feats = {}
    for name, prompt in PROMPTS.items():
        inputs = clap.processor(text=[prompt], return_tensors="pt")
        inputs = {k: v.to(clap.device) for k, v in inputs.items()}
        with torch.no_grad():
            feat = clap.model.get_text_features(**inputs).squeeze(0).float().cpu().numpy()
        text_feats[name] = feat / (np.linalg.norm(feat) + 1e-12)

    def rank(audio_48k: np.ndarray) -> dict[str, float]:
        emb = clap.embed(audio_48k)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return {name: float(np.dot(emb, vec)) for name, vec in text_feats.items()}

    entries: list[tuple[str, Path]] = []
    source = args.source or (
        Path("/Users/kkszhang/project/毕设")
        / "experiments"
        / "cover_test_kit"
        / "sources"
        / "chinese_female_120s.wav"
    )
    if source.is_file():
        entries.append(("00_SOURCE", source))
    for arm_dir in sorted(p for p in args.dir.iterdir() if p.is_dir()):
        wavs = sorted(arm_dir.glob("*.wav"))
        if wavs:
            entries.append((arm_dir.name, wavs[0]))

    if not entries:
        print("no audio found", file=sys.stderr)
        return 1

    rows = []
    for name, wav in entries:
        audio, rate = load_audio(wav)
        mono = to_mono_48k(audio, rate)
        row = {"arm": name, "wav": str(wav)}
        row["all"] = rank(mono)
        vocal = segment_audio(mono, vocal=True, sample_rate=48000)
        bg = segment_audio(mono, vocal=False, sample_rate=48000)
        if vocal.size:
            row["vocal"] = rank(vocal)
        if bg.size:
            row["bg"] = rank(bg)
        rows.append(row)
        order = sorted(row["all"].items(), key=lambda kv: -kv[1])
        print(
            f"[{name}] "
            + "  ".join(f"{k}={v:.4f}" for k, v in order)
            + f"   (bg: "
            + "  ".join(f"{k}={v:.4f}" for k, v in sorted(row.get('bg', {}).items(), key=lambda kv: -kv[1]))
            + ")",
            flush=True,
        )

    args.out.write_text(json.dumps({"prompts": PROMPTS, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
