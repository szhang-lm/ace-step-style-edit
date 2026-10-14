#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 04_corrected_covers 的 12 个 Cover 音频跑 Qwen3-ASR，只取文本（不加载对齐器）。"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path('/cfs/cfs-h80qwxi5/kkszhang')
ASR = ROOT / 'FunASR/models/qwen3/Qwen/Qwen3-ASR-1___7B'

from qwen_asr import Qwen3ASRModel


def lang_for(name: str) -> str:
    n = name.lower()
    if 'chinese' in n:
        return 'Chinese'
    if 'english' in n:
        return 'English'
    return None


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / 'song_gene_and_edit/ACE-Step-1.5/outputs/multi_song_test/04_corrected_covers')
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / '05_asr_covers'
    gpu = sys.argv[3] if len(sys.argv) > 3 else 'cuda:0'
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(src.glob('*.wav'))
    if not wavs:
        print('no wav in', src, flush=True)
        return 1

    print(f'loading Qwen3-ASR on {gpu} ...', flush=True)
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(
        str(ASR), dtype=torch.bfloat16, device_map=gpu,
        low_cpu_mem_usage=True, max_new_tokens=2048,
    )
    print(f'loaded in {time.time()-t0:.1f}s', flush=True)

    for wav in wavs:
        lang = lang_for(wav.name)
        print(f'[{wav.name}] lang={lang}', flush=True)
        t = time.time()
        try:
            results = model.transcribe(audio=[str(wav)], language=lang)
            r = results[0]
            text = r.text
            out_json = out_dir / (wav.stem + '.json')
            out_json.write_text(json.dumps({
                'wav': str(wav), 'requested_language': lang,
                'language': r.language, 'text': text,
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'  [{time.time()-t:.1f}s] chars={len(text)} saved={out_json.name}', flush=True)
        except Exception as exc:
            print(f'  FAIL {wav.name}: {exc!r}', flush=True)

    print('DONE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
