#!/usr/bin/env python3
"""Aggregate the multi-seed cns=0.0 ablation into arm-level statistics.

Primary endpoint: lyric coverage from Qwen3-ASR (stable to ~2 points across
replicates).  Secondary endpoint: CLAP style distance, which has a much wider
run-to-run spread and is therefore only reported as a mean +- std sanity check
that protecting the vocals does not collapse the style change.

Usage:
    python aggregate_ablation.py [--outputs DIR] [--json OUT]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

CJK = re.compile(r"[\u4e00-\u9fff]")

REPO_ROOT = Path(__file__).resolve().parent.parent
THESIS_ROOT = REPO_ROOT.parent
OUTPUTS = THESIS_ROOT / "experiments" / "nonuniform_noise" / "outputs"
LYRICS = THESIS_ROOT / "experiments" / "cover_test_kit" / "03_corrected_lyrics" / "chinese_female.txt"
SOURCE = THESIS_ROOT / "experiments" / "cover_test_kit" / "sources" / "chinese_female_120s.wav"

ARMS = ("k0", "k1", "k3", "k8")
ARM_LABEL = {
    "k0": "无保护 (基线)",
    "k1": "仅初始非均匀, 1 步",
    "k3": "钳制 3 步",
    "k8": "钳制全程",
}


def reference_lines() -> list[str]:
    """Return the lyric lines that were actually sung, markers stripped."""
    lines = []
    for raw in LYRICS.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if text and not text.startswith("["):
            lines.append(text)
    return lines


def reference_tokens() -> list[str]:
    """Return the character tokens of the lyrics that were actually sung."""
    return CJK.findall(LYRICS.read_text(encoding="utf-8"))


def worst_line_loss(reference_lines: list[str], matched_flags: list[bool]) -> float:
    """Return the largest per-line fraction of dropped characters.

    Aggregate coverage hides *where* the errors fall: two runs can both score
    ~90% while one drops a whole line and the other spreads small errors evenly,
    and listeners hear the first as broken.  This makes that visible.
    """
    idx = 0
    worst = 0.0
    for line in reference_lines:
        count = len(CJK.findall(line))
        if not count:
            continue
        segment = matched_flags[idx : idx + count]
        idx += count
        if len(segment) < count:
            break
        lost = sum(1 for flag in segment if not flag)
        worst = max(worst, lost / count)
    return worst


def coverage(reference: list[str], hypothesis: str) -> tuple[float, int]:
    """Return (coverage, matched) of ``hypothesis`` against ``reference``."""
    hyp = CJK.findall(hypothesis)
    matcher = SequenceMatcher(a=reference, b=hyp, autojunk=False)
    matched_flags = [False] * len(reference)
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                matched_flags[k] = True
    matched = sum(matched_flags)
    return (matched / len(reference) if reference else 0.0), matched, matched_flags


def collect_targets(outputs: Path) -> list[tuple[str, int, str, Path]]:
    """Return ``(arm, seed, style, wav)`` for every generated cover."""
    targets = []
    for manifest_path in sorted(outputs.glob("ablation*/**/manifest_*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        arm = manifest_path.parent.name
        if arm not in ARMS:
            continue
        seed = int(manifest.get("seed", 0))
        style = manifest.get("style", "?")
        for audio in manifest.get("audios", []):
            wav = Path(audio["path"])
            if wav.is_file():
                targets.append((arm, seed, style, wav))
    return targets


def run_asr(targets: list[tuple[str, int, str, Path]], reference: list[str]) -> dict:
    """Transcribe every target once and return coverage per (arm, seed, style)."""
    import torch
    from qwen_asr import Qwen3ASRModel

    model_dir = THESIS_ROOT / "experiments" / "eval_models" / "Qwen3-ASR-1.7B"
    print(f"loading Qwen3-ASR from {model_dir} ...", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        str(model_dir), dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    )
    lyric_lines = reference_lines()
    results = {}
    for arm, seed, style, wav in targets:
        key = (arm, seed, style)
        try:
            out = model.transcribe(audio=str(wav), language="Chinese", return_time_stamps=False)
            text = out[0].text if isinstance(out, (list, tuple)) else out.text
        except Exception as exc:  # noqa: BLE001 - record and continue
            print(f"  ASR failed for {key}: {exc}", flush=True)
            continue
        cov, matched, flags = coverage(reference, text)
        worst = worst_line_loss(lyric_lines, flags)
        results[key] = {
            "coverage": cov,
            "matched": matched,
            "worst_line_loss": worst,
            "text": text,
        }
        print(
            f"  s{seed} {arm}/{style:<6} coverage={cov*100:5.1f}% "
            f"worst_line_loss={worst*100:5.1f}% matched={matched:3d}",
            flush=True,
        )
    return results


def summarise(rows: dict) -> dict:
    """Aggregate per-arm means and standard deviations."""
    by_arm: dict[str, list[float]] = defaultdict(list)
    by_arm_style: dict[tuple[str, str], list[float]] = defaultdict(list)
    worst_by_arm: dict[str, list[float]] = defaultdict(list)
    for (arm, seed, style), value in rows.items():
        by_arm[arm].append(value["coverage"])
        by_arm_style[(arm, style)].append(value["coverage"])
        if value.get("worst_line_loss") is not None:
            worst_by_arm[arm].append(value["worst_line_loss"])

    def stats(values: list[float]) -> tuple[float, float, int]:
        if not values:
            return (float("nan"), float("nan"), 0)
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return mean, var ** 0.5, len(values)

    summary = {}
    for arm in ARMS:
        mean, std, n = stats(by_arm.get(arm, []))
        w_mean, w_std, _ = stats(worst_by_arm.get(arm, []))
        summary[arm] = {
            "mean": mean,
            "std": std,
            "n": n,
            "worst_line_mean": w_mean,
            "worst_line_std": w_std,
            "by_style": {},
        }
        for style in ("jazz", "rock"):
            m, s, cnt = stats(by_arm_style.get((arm, style), []))
            summary[arm]["by_style"][style] = {"mean": m, "std": s, "n": cnt}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    parser.add_argument("--json", type=Path, default=OUTPUTS / "ablation_seeds_asr.json")
    parser.add_argument("--from-json", type=Path, help="skip ASR, reuse a previous run")
    args = parser.parse_args()

    reference = reference_tokens()
    targets = collect_targets(args.outputs)
    if not targets:
        print("no ablation outputs found", file=sys.stderr)
        return 1
    print(f"{len(targets)} covers across {len({t[1] for t in targets})} seeds\n")

    if args.from_json and args.from_json.is_file():
        raw = json.loads(args.from_json.read_text(encoding="utf-8"))
        rows = {
            (r["arm"], r["seed"], r["style"]): r for r in raw.get("rows", [])
        }
    else:
        rows = run_asr(targets, reference)
        payload_rows = [
            {"arm": a, "seed": s, "style": st, **v} for (a, s, st), v in rows.items()
        ]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"rows": payload_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        f"\n{'arm':<6}{'说明':<22}{'mean':>9}{'std':>8}"
        f"{'最差句丢字':>11}{'n':>4}   jazz / rock"
    )
    print("-" * 90)
    summary = summarise(rows)
    for arm in ARMS:
        entry = summary[arm]
        if not entry["n"]:
            continue
        by_style = " / ".join(
            f"{entry['by_style'][s]['mean']*100:.1f}%"
            for s in ("jazz", "rock")
            if entry["by_style"][s]["n"]
        )
        print(
            f"{arm:<6}{ARM_LABEL[arm]:<22}{entry['mean']*100:>8.1f}%{entry['std']*100:>7.1f}"
            f"{entry['worst_line_mean']*100:>10.1f}%{entry['n']:>4}   {by_style}"
        )

    base = summary["k0"]
    if base["n"]:
        print(f"\n相对 k0 基线的提升 (k0 = {base['mean']*100:.1f}%):")
        for arm in ARMS[1:]:
            entry = summary[arm]
            if entry["n"]:
                print(
                    f"  {arm}: {entry['mean']*100:5.1f}%  "
                    f"(+{(entry['mean']-base['mean'])*100:.1f} 个百分点)"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
