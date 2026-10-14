#!/usr/bin/env python3
"""Build the markdown report for the non-uniform noise ablation.

Reads the evaluation JSON produced by ``evaluate_nonuniform_noise.py`` and prints
a table plus the H1 / H2 verdict from the plan document.

Usage:
    python make_report.py --eval <evaluation.json> [--out report.md]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional


ARM_DESCRIPTIONS = {
    "fidelity_ref": "A | 全局 cns=0.6 (noise_t=0.375),无保护 —— 歌词保真参照",
    "global_075": "B1 | 全局 cns=0.25 (noise_t=0.75),无保护",
    "global_100": "B2 | 全局 cns=0.0 (noise_t=1.00),无保护 —— 最大曲风变化",
    "init1": "C1 | 非均匀初始噪声,仅第 1 步 (k=1)",
    "cont": "C2 | 源轨迹钳制到人声下限 t=0.375 (k=3)",
    "init1_cont": "C3 | 初始 + 持续钳制 (k=3,与 C2 同轨迹,确定性复现检查)",
    "init3": "C4 | 超大初始窗 (k=3,过保护上界)",
}

H1_COVERAGE_RATIO = 0.90  # C 组歌词覆盖率需 >= A 组的 90%
H1_STYLE_RATIO = 1.50  # C 组曲风差异度需 >= A 组的 1.5 倍


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("rows", [])


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    return grouped


def mean(values: list[float]) -> Optional[float]:
    usable = [v for v in values if v is not None and v == v]
    return sum(usable) / len(usable) if usable else None


def fmt(value: Optional[float], digits: int = 4, scale: float = 1.0) -> str:
    if value is None:
        return "-"
    return f"{value * scale:.{digits}f}"


def verdict(grouped: dict[str, list[dict]]) -> list[str]:
    """Return the H1 / H2 verdict lines.

    The style axis uses CLAP: MERT's pooled embedding barely separates the source
    from a genuine rock re-arrangement, so it cannot support a 1.5x criterion.
    """
    lines: list[str] = []

    def agg(group: str, key: str) -> Optional[float]:
        return mean([r.get(key) for r in grouped.get(group, [])])

    ref_cov, ref_style = agg("fidelity_ref", "coverage"), agg("fidelity_ref", "clap_all")
    lines.append("### 判据")
    lines.append("")
    lines.append(
        f"- H1 要求实验组 `歌词覆盖率 >= {H1_COVERAGE_RATIO:.0%} × A组` 且 "
        f"`曲风差异度 >= {H1_STYLE_RATIO:.1f} × A组`。"
    )
    if ref_cov is None or ref_style is None:
        lines.append("- 参照组 A 的指标缺失,无法判定(需先跑 `fidelity_ref`)。")
        return lines
    lines.append(
        f"- A 组参照:覆盖率 {ref_cov*100:.1f}%,曲风差异度(全局) {ref_style:.4f}。"
    )
    lines.append("")
    lines.append(
        f"| 组 | 覆盖率 | 覆盖率/A | 曲风差异(全局) | 差异/A | 曲风差异(伴奏) | H1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for group in ARM_DESCRIPTIONS:
        rows = grouped.get(group)
        if not rows:
            continue
        cov, style, style_bg = (
            agg(group, "coverage"),
            agg(group, "clap_all"),
            agg(group, "clap_background"),
        )
        cov_ratio = cov / ref_cov if cov is not None and ref_cov else None
        style_ratio = style / ref_style if style is not None and ref_style else None
        ok = (
            "✅ 成立"
            if cov_ratio is not None
            and style_ratio is not None
            and cov_ratio >= H1_COVERAGE_RATIO
            and style_ratio >= H1_STYLE_RATIO
            else "❌ 不成立"
        )
        lines.append(
            f"| {group} | {fmt(cov, 1, 100)}% | {fmt(cov_ratio, 3)} | "
            f"{fmt(style)} | {fmt(style_ratio, 3)} | {fmt(style_bg)} | {ok} |"
        )

    # H2: does the accompaniment's style change survive next to protected vocals?
    lines.append("")
    lines.append("### H2(attention 污染)检查")
    lines.append("")
    ref_bg = agg("fidelity_ref", "clap_background")
    for group in ("init1", "cont", "init3"):
        rows = grouped.get(group)
        if not rows:
            continue
        bg = agg(group, "clap_background")
        voc_style = agg(group, "clap_vocal")
        ref_voc_style = agg("fidelity_ref", "clap_vocal")
        lines.append(
            f"- `{group}`:伴奏区曲风差异 {fmt(bg)}"
            + (f"(A组 {fmt(ref_bg)},比值 {fmt(bg/ref_bg,3)})" if ref_bg else "")
            + f";人声区曲风差异 {fmt(voc_style)}"
            + (
                f"(A组 {fmt(ref_voc_style)},比值 {fmt(voc_style/ref_voc_style,3)})"
                if ref_voc_style
                else ""
            )
        )
    lines.append("")
    lines.append(
        "若某组在伴奏区拿到了明显更大的曲风差异,同时人声区差异没有同步变大,"
        "说明分区控制生效、未见人声被伴奏带崩(H2 暂不成立);反之则说明存在污染。"
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--title", default="非均匀噪声曲风编辑 —— 消融实验结果")
    args = parser.parse_args()

    rows = load_rows(args.eval)
    grouped = group_rows(rows)

    lines = [f"# {args.title}", ""]
    lines.append(f"评测数据:`{args.eval}`")
    lines.append("")
    lines.append("## 逐条结果")
    lines.append("")
    lines.append(
        "| 组 | 音频 | 覆盖率 | 丢字 | 多余 | CLAP(全局) | CLAP(人声区) | CLAP(伴奏区) | CLAP音色(人声) | MERT(全局) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: (r["group"], r["wav"])):
        lines.append(
            f"| {row['group']} | {Path(row['wav']).name[:12]} | "
            f"{fmt(row.get('coverage'), 1, 100)}% | {row.get('dropped', '-')} | "
            f"{row.get('extra', '-')} | {fmt(row.get('clap_all'))} | "
            f"{fmt(row.get('clap_vocal'))} | {fmt(row.get('clap_background'))} | "
            f"{fmt(row.get('clap_timbre_vocal'))} | {fmt(row.get('style_all'))} |"
        )

    lines.append("")
    lines.append("## 分组均值")
    lines.append("")
    lines.append(
        "| 组 | 说明 | 覆盖率 | CLAP(全局) | CLAP(人声区) | CLAP(伴奏区) | CLAP音色(人声) | MERT(全局) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for group, description in ARM_DESCRIPTIONS.items():
        if group not in grouped:
            continue
        lines.append(
            f"| {group} | {description} | "
            f"{fmt(mean([r.get('coverage') for r in grouped[group]]), 1, 100)}% | "
            f"{fmt(mean([r.get('clap_all') for r in grouped[group]]))} | "
            f"{fmt(mean([r.get('clap_vocal') for r in grouped[group]]))} | "
            f"{fmt(mean([r.get('clap_background') for r in grouped[group]]))} | "
            f"{fmt(mean([r.get('clap_timbre_vocal') for r in grouped[group]]))} | "
            f"{fmt(mean([r.get('style_all') for r in grouped[group]]))} |"
        )

    lines.append("")
    lines.extend(verdict(grouped))
    lines.append("")

    text = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
