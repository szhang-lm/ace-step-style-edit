#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 pipeline 的 Qwen3-ASR + ForcedAligner 识别歌曲歌词并输出逐字时间戳。

输出每个音频一个 json：text（全文）、tokens（逐字 start/end 秒）、
lines（按句号/问号/感叹号粗分的歌词行及起止秒）。
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path('/cfs/cfs-h80qwxi5/kkszhang')
ASR = ROOT / 'FunASR/models/qwen3/Qwen/Qwen3-ASR-1___7B'
ALIGNER = ROOT / 'FunASR/models/qwen3/Qwen/Qwen3-ForcedAligner-0___6B'

from qwen_asr import Qwen3ASRModel

LINE_END = set('。！？!?')


def lang_for(name: str) -> str:
    n = name.lower()
    if 'chinese' in n:
        return 'Chinese'
    if 'english' in n:
        return 'English'
    return None  # 让模型自动判断


def token_spans(text: str, tokens: list) -> list:
    """把每个 token 定位到原文字符区间，返回 [(start_char, end_char), ...] 或 None。"""
    spans = []
    cursor = 0
    lower = text.casefold()
    for tok in tokens:
        target = tok['text'].casefold()
        pos = lower.find(target, cursor)
        if pos < 0:
            pos = lower.find(target)
            if pos < cursor:
                return None
        spans.append((pos, pos + len(target)))
        cursor = pos + len(target)
    return spans


def build_lines(text: str, tokens: list, spans: list) -> list:
    """按句末标点粗分歌词行，返回 [{text, start, end}]。"""
    lines = []
    cur = []
    for i, tok in enumerate(tokens):
        cur.append(tok)
        _, end_char = spans[i]
        # 当前 token 之后是否跟句末标点（含 token 自身结尾）
        seg = text[max(0, end_char):end_char + 1]
        if seg and seg[0] in LINE_END:
            lines.append({
                'text': text[spans[0][0]:spans[i][1]],
                'start': cur[0]['start'],
                'end': tok['end'],
            })
            cur = []
    if cur:
        lines.append({
            'text': text[spans[-1][0]:spans[-1][1]] if lines else text,
            'start': cur[0]['start'],
            'end': cur[-1]['end'],
        })
    return lines


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / 'song_gene_and_edit/ACE-Step-1.5/outputs/multi_song_test/sources')
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / '02_asr_aligned'
    gpu = sys.argv[3] if len(sys.argv) > 3 else 'cuda:0'
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(src.glob('*.wav'))
    if not wavs:
        print('no wav in', src, flush=True)
        return 1

    print(f'loading Qwen3-ASR + ForcedAligner on {gpu} ...', flush=True)
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(
        str(ASR),
        forced_aligner=str(ALIGNER),
        forced_aligner_kwargs={
            'dtype': torch.bfloat16,
            'device_map': gpu,
            'low_cpu_mem_usage': True,
        },
        dtype=torch.bfloat16,
        device_map=gpu,
        low_cpu_mem_usage=True,
        max_new_tokens=2048,
    )
    print(f'loaded in {time.time()-t0:.1f}s', flush=True)

    for wav in wavs:
        lang = lang_for(wav.name)
        tag = lang or 'auto'
        print(f'\n===== {wav.name}  (language={tag}) =====', flush=True)
        t = time.time()
        try:
            results = model.transcribe(
                audio=[str(wav)],
                language=lang,
                return_time_stamps=True,
            )
            r = results[0]
            text = r.text
            tokens = [{'text': x.text, 'start': round(x.start_time, 3), 'end': round(x.end_time, 3)}
                      for x in list(r.time_stamps.items)] if r.time_stamps is not None else []
            spans = token_spans(text, tokens)
            lines = build_lines(text, tokens, spans) if spans is not None else []
            print(f'[{time.time()-t:.1f}s] lang={r.language}  tokens={len(tokens)}\n{text}', flush=True)
            out_json = out_dir / (wav.stem + '.json')
            out_json.write_text(json.dumps({
                'wav': str(wav),
                'requested_language': lang,
                'language': r.language,
                'text': text,
                'tokens': tokens,
                'lines': lines,
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'saved -> {out_json}', flush=True)
        except Exception as exc:
            print(f'FAIL {wav.name}: {exc!r}', flush=True)

    print('\nDONE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
