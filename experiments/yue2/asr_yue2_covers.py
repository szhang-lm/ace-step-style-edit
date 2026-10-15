#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 YuE2 的 12 个 cover (audio.flac) 跑 Qwen3-ASR，只取文本。

用法:
    python asr_yue2_covers.py [gpu] [start_index]
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path('/cfs/cfs-h80qwxi5/kkszhang')
ASR = ROOT / 'FunASR/models/qwen3/Qwen/Qwen3-ASR-1___7B'
YUE2 = ROOT / 'song_gene_and_edit/YuE2/outputs'

# (目录名, 歌, 曲风, 语言)
COVERS = [
    ('cover_jazz',              'chinese_female', 'jazz',       'Chinese'),
    ('cover_rock',              'chinese_female', 'rock',       'Chinese'),
    ('cover_electronic',        'chinese_female', 'electronic', 'Chinese'),
    ('cover_folk',              'chinese_female', 'folk',       'Chinese'),
    ('english_female_jazz',     'english_female', 'jazz',       'English'),
    ('english_female_rock',     'english_female', 'rock',       'English'),
    ('english_female_electronic','english_female','electronic', 'English'),
    ('english_female_folk',     'english_female', 'folk',       'English'),
    ('english_male_jazz',       'english_male',   'jazz',       'English'),
    ('english_male_rock',       'english_male',   'rock',       'English'),
    ('english_male_electronic', 'english_male',   'electronic', 'English'),
    ('english_male_folk',       'english_male',   'folk',       'English'),
]

from qwen_asr import Qwen3ASRModel


def main():
    gpu = sys.argv[1] if len(sys.argv) > 1 else 'cuda:0'
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out_dir = ROOT / 'song_gene_and_edit/YuE2/outputs/05_asr_yue2'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'loading Qwen3-ASR on {gpu} ...', flush=True)
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(
        str(ASR), dtype=torch.bfloat16, device_map=gpu,
        low_cpu_mem_usage=True, max_new_tokens=2048,
    )
    print(f'loaded in {time.time()-t0:.1f}s', flush=True)

    for idx, (subdir, song, style, lang) in enumerate(COVERS):
        if idx < start:
            continue
        wav = YUE2 / subdir / 'audio.flac'
        if not wav.exists():
            print(f'[{subdir}] MISSING {wav}', flush=True)
            continue
        print(f'[{idx}/{len(COVERS)}] {subdir} lang={lang}', flush=True)
        t = time.time()
        try:
            results = model.transcribe(audio=[str(wav)], language=lang)
            r = results[0]
            text = r.text
            out_json = out_dir / f'{song}_{style}.json'
            out_json.write_text(json.dumps({
                'wav': str(wav), 'requested_language': lang,
                'language': r.language, 'text': text,
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'  [{time.time()-t:.1f}s] chars={len(text)} saved={out_json.name}', flush=True)
        except Exception as exc:
            print(f'  FAIL {subdir}: {exc!r}', flush=True)

    print('DONE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
