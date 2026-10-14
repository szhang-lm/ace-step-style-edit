#!/usr/bin/env bash
# Compare three mask definitions at the SAME light protection setting.
#
# The source stays the FULL MIX in every arm, so the arrangement template is
# preserved (feeding a vocals-only stem removed that template and made the new
# accompaniment sound thin).  Only the mask definition changes:
#
#   asr_win1s  ASR lyric line + first 1.0s window  (current best, 6.7% of frames)
#   stem_f0    F0-based mask from the separated vocal stem (48.7% of frames)
#
# The stem mask marks where the voice actually is rather than where the lyrics
# say a line exists, so it should keep the words without anchoring the silence
# between them.
set -u
REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
MASK="$THESIS_ROOT/experiments/cover_test_kit/02_asr_aligned/chinese_female_120s.json"
NPY="$THESIS_ROOT/experiments/nonuniform_noise/masks/vocal_mask_stem_3000.npy"
SRC="$THESIS_ROOT/experiments/cover_test_kit/sources/chinese_female_120s.wav"
OUT="$THESIS_ROOT/experiments/nonuniform_noise/outputs/vocmask_test"
PY="$REPO_ROOT/.venv/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACESTEP_VOCAL_MASK_PATH="$MASK"
export ACESTEP_VOCAL_GRANULARITY=line
export ACESTEP_VOCAL_MASK_FADE_SEC=0.25
export ACESTEP_VOCAL_MODE=init
export ACESTEP_VOCAL_INIT_STEPS=1
export ACESTEP_VOCAL_CNS=0.375
mkdir -p "$OUT"

echo "=== asr_win1s ==="
ACESTEP_VOCAL_ONSET_SECONDS=1.0 env -u ACESTEP_VOCAL_MASK_NPY \
"$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
  --style jazz --cover-noise-strength 0.0 --seed 20264101 --source "$SRC" \
  --vocal-mask "$MASK" --vocal-cns 0.375 --source-as-reference \
  --output-dir "$OUT/asr_win1s" > "$OUT/asr_win1s.log" 2>&1
echo "rc=$?"

echo "=== stem_f0 ==="
ACESTEP_VOCAL_MASK_NPY="$NPY" env -u ACESTEP_VOCAL_ONSET_SECONDS \
"$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
  --style jazz --cover-noise-strength 0.0 --seed 20264101 --source "$SRC" \
  --vocal-mask "$MASK" --vocal-cns 0.375 --source-as-reference \
  --output-dir "$OUT/stem_f0" > "$OUT/stem_f0.log" 2>&1
echo "rc=$?"

echo "=== vocmask test finished ==="
