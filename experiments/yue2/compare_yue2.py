#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YuE2 cover 歌词覆盖率：cover ASR vs 喂进去的歌词（YuE2 是重唱，歌词是输入）。

中文喂的是 sources/chinese_female_lyrics.txt（完整版）；
英文喂的是 03_corrected_lyrics 里的英文歌词。
"""
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path('/cfs/cfs-h80qwxi5/kkszhang/song_gene_and_edit')
MT = ROOT / 'ACE-Step-1.5/outputs/multi_song_test'
ASR_DIR = ROOT / 'YuE2/outputs/05_asr_yue2'
CJK = re.compile(r'[\u4e00-\u9fff]')

LYRICS = {
    'chinese_female': MT / '03_corrected_lyrics/chinese_female.txt',
    'english_female': MT / '03_corrected_lyrics/english_female.txt',
    'english_male': MT / '03_corrected_lyrics/english_male.txt',
}
LANGS = {'chinese_female': 'zh', 'english_female': 'en', 'english_male': 'en'}
STYLES = ['jazz', 'rock', 'electronic', 'folk']


def norm_zh(s):
    return CJK.findall(s)


def norm_en(s):
    s = s.lower().replace('\u2019', "'")
    s = re.sub(r"[^a-z0-9'\s]", ' ', s)
    return s.split()


def load_ref(path, lang):
    toks = []
    for ln in path.read_text(encoding='utf-8').splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('['):
            continue
        t = norm_zh(ln) if lang == 'zh' else norm_en(ln)
        toks.extend(t)
    return toks


def compare(ref, asr):
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
    print('=' * 90)
    print(f"{'歌':<16}{'曲风':<12}{'歌词':>5}{'识别':>5}{'匹配':>5}{'丢字':>5}{'多余':>5}{'覆盖率':>8}")
    print('-' * 90)
    for song, lang in LANGS.items():
        ref_toks = load_ref(LYRICS[song], lang)
        for style in STYLES:
            j = ASR_DIR / f'{song}_{style}.json'
            if not j.exists():
                print(f'{song:<16}{style:<12}  (缺 json)')
                continue
            asr_toks = norm_zh(json.loads(j.read_text())['text']) if lang == 'zh' \
                else norm_en(json.loads(j.read_text())['text'])
            matched, dropped, extra = compare(ref_toks, asr_toks)
            cov = matched / len(ref_toks) if ref_toks else 0.0
            print(f'{song:<16}{style:<12}{len(ref_toks):>5}{len(asr_toks):>5}'
                  f'{matched:>5}{dropped:>5}{extra:>5}{cov*100:>7.1f}%')
    print('\nDONE')


if __name__ == '__main__':
    main()
