#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 Cover 的 ASR 文本 与 参考音频的 ASR 文本（原歌词基准）。

覆盖率 = 参考 ASR 中被 Cover ASR 覆盖的比例；丢字 = 参考有而 Cover 无；
多余 = Cover 有而参考无（重复/幻觉）。中文按字、英文按词对齐。
"""
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path('/cfs/cfs-h80qwxi5/kkszhang/song_gene_and_edit/ACE-Step-1.5/outputs/multi_song_test')
CJK = re.compile(r'[\u4e00-\u9fff]')

SONGS = [
    ('chinese_female', 'chinese_female_120s', 'zh'),
    ('english_female', 'english_female_80s', 'en'),
    ('english_male', 'english_male_120s', 'en'),
]
STYLES = ['jazz', 'rock', 'electronic', 'folk']


def norm_zh(s: str) -> list:
    return CJK.findall(s)


def norm_en(s: str) -> list:
    s = s.lower().replace('’', "'")
    s = re.sub(r"[^a-z0-9'\s]", ' ', s)
    return s.split()


def compare(ref: list, asr: list):
    sm = SequenceMatcher(a=ref, b=asr, autojunk=False)
    ref_match = [False] * len(ref)
    asr_match = [False] * len(asr)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i1, i2):
                ref_match[k] = True
            for k in range(j1, j2):
                asr_match[k] = True
    matched = sum(ref_match)
    return matched, len(ref) - matched, len(asr) - sum(asr_match)


def main():
    asr_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / '05_asr_covers'
    ref_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / '02_asr_aligned'

    print('=' * 100)
    print(f"{'歌':<16}{'曲风':<12}{'参考':>5}{'Cover':>5}{'匹配':>5}{'丢字':>5}{'多余':>5}{'覆盖率':>8}")
    print('-' * 100)
    for song, src_name, lang in SONGS:
        ref = json.loads((ref_dir / f'{src_name}.json').read_text())['text']
        ref_toks = norm_zh(ref) if lang == 'zh' else norm_en(ref)
        for style in STYLES:
            j = asr_dir / f'{song}_{style}.json'
            if not j.exists():
                print(f'{song:<16}{style:<12}  (缺 json)')
                continue
            asr = json.loads(j.read_text())['text']
            asr_toks = norm_zh(asr) if lang == 'zh' else norm_en(asr)
            matched, dropped, extra = compare(ref_toks, asr_toks)
            cov = matched / len(ref_toks) if ref_toks else 0.0
            print(f'{song:<16}{style:<12}{len(ref_toks):>5}{len(asr_toks):>5}'
                  f'{matched:>5}{dropped:>5}{extra:>5}{cov*100:>7.1f}%')
    print('\nDONE')


if __name__ == '__main__':
    main()
