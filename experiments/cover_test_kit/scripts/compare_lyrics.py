#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 Cover 的 ASR 文本与校正歌词，检测丢字/重复/错序。

用法:
    python compare_lyrics.py [asr_dir] [lyrics_dir]

输出:
    - 汇总表（歌 × 曲风：参考长度、识别长度、匹配、丢字、多余、覆盖率）
    - 每行歌词的丢失情况
"""
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path('/cfs/cfs-h80qwxi5/kkszhang/song_gene_and_edit/ACE-Step-1.5/outputs/multi_song_test')
CJK = re.compile(r'[\u4e00-\u9fff]')


def norm_zh(s: str) -> list:
    return CJK.findall(s)


def norm_en(s: str) -> list:
    s = s.lower().replace('’', "'")
    s = re.sub(r"[^a-z0-9'\s]", ' ', s)
    return s.split()


def load_ref(path: Path, lang: str):
    """返回 (lines, per-line token lists)。跳过 [标记] 行。"""
    lines = []
    toks_per_line = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('['):
            continue
        t = norm_zh(ln) if lang == 'zh' else norm_en(ln)
        if t:
            lines.append(ln)
            toks_per_line.append(t)
    return lines, toks_per_line


def compare(ref_toks, asr_toks):
    sm = SequenceMatcher(a=ref_toks, b=asr_toks, autojunk=False)
    opcodes = sm.get_opcodes()
    ref_match = [False] * len(ref_toks)
    asr_match = [False] * len(asr_toks)
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for k in range(i1, i2):
                ref_match[k] = True
            for k in range(j1, j2):
                asr_match[k] = True
    matched = sum(ref_match)
    dropped = len(ref_toks) - matched
    extra = len(asr_toks) - sum(asr_match)
    return ref_match, asr_match, matched, dropped, extra


def main():
    asr_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / '05_asr_covers'
    lyr_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / '03_corrected_lyrics'

    songs = {
        'chinese_female': 'zh',
        'english_female': 'en',
        'english_male': 'en',
    }
    styles = ['jazz', 'rock', 'electronic', 'folk']

    print('=' * 110)
    print(f"{'歌':<16}{'曲风':<12}{'参考':>5}{'识别':>5}{'匹配':>5}{'丢字':>5}{'多余':>5}{'覆盖率':>8}")
    print('-' * 110)
    for song, lang in songs.items():
        lines, toks_per_line = load_ref(lyr_dir / f'{song}.txt', lang)
        ref_toks = [t for tl in toks_per_line for t in tl]
        for style in styles:
            j = asr_dir / f'{song}_{style}.json'
            if not j.exists():
                print(f'{song:<16}{style:<12}  (缺 json)')
                continue
            data = json.loads(j.read_text())
            asr_toks = norm_zh(data['text']) if lang == 'zh' else norm_en(data['text'])
            ref_match, asr_match, matched, dropped, extra = compare(ref_toks, asr_toks)
            cov = matched / len(ref_toks) if ref_toks else 0.0
            print(f'{song:<16}{style:<12}{len(ref_toks):>5}{len(asr_toks):>5}'
                  f'{matched:>5}{dropped:>5}{extra:>5}{cov*100:>7.1f}%')

    # 每行丢失详情
    print('\n' + '=' * 110)
    print('每行歌词丢失详情（丢字占比 > 10% 的行）')
    print('=' * 110)
    for song, lang in songs.items():
        lines, toks_per_line = load_ref(lyr_dir / f'{song}.txt', lang)
        for style in styles:
            j = asr_dir / f'{song}_{style}.json'
            if not j.exists():
                continue
            data = json.loads(j.read_text())
            asr_toks = norm_zh(data['text']) if lang == 'zh' else norm_en(data['text'])
            ref_toks = [t for tl in toks_per_line for t in tl]
            ref_match, _, _, _, _ = compare(ref_toks, asr_toks)
            # 把 ref_match 切回每行
            idx = 0
            bad = []
            for ln, tl in zip(lines, toks_per_line):
                seg = ref_match[idx:idx + len(tl)]
                lost = sum(1 for m in seg if not m)
                frac = lost / len(tl)
                if frac > 0.10:
                    lost_toks = [t for t, m in zip(tl, seg) if not m]
                    bad.append((frac, ln, ''.join(lost_toks) if lang == 'zh' else ' '.join(lost_toks)))
                idx += len(tl)
            if bad:
                print(f'\n[{song}/{style}]')
                for frac, ln, lost_toks in sorted(bad, reverse=True):
                    print(f'  丢{frac*100:.0f}%  {ln}   <-- 缺: {lost_toks}')

    print('\nDONE')


if __name__ == '__main__':
    main()
