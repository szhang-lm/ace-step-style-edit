#!/bin/bash
set -e
cd /cfs/cfs-h80qwxi5/kkszhang/song_gene_and_edit/YuE2
source .venv/bin/activate

LYRICS=/cfs/cfs-h80qwxi5/kkszhang/song_gene_and_edit/ACE-Step-1.5/outputs/multi_song_test/sources/chinese_female_lyrics.txt

# 生成 3 个 request 文件（jazz 已单独完成）
python - <<'PY'
import json
from pathlib import Path
lyrics = Path("/cfs/cfs-h80qwxi5/kkszhang/song_gene_and_edit/ACE-Step-1.5/outputs/multi_song_test/sources/chinese_female_lyrics.txt").read_text(encoding="utf-8")
styles = {
    "rock": "Chinese, rock, powerful female vocal, electric guitar, bass and drums, driving rhythm, energetic",
    "electronic": "Chinese, electronic, synth-pop, female vocal, synthesizer, electronic beats, atmospheric and modern",
    "folk": "Chinese, folk, gentle female vocal, acoustic guitar, soft percussion, warm and intimate",
}
for genre, style in styles.items():
    req = {"style": style, "lyrics": lyrics, "cot": "melody", "seed": 42}
    Path(f"cover-request-{genre}.json").write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written cover-request-{genre}.json")
PY

for genre in rock electronic folk; do
  echo "===== [$(date '+%F %T')] 生成 $genre cover ====="
  python examples/generate.py --request "cover-request-${genre}.json" \
    --abc-file cover-score/score.abc --cot melody --output "outputs/cover_${genre}" 2>&1 | tail -3
  echo "===== $genre 完成, exit=$? ====="
done
echo "全部完成"
