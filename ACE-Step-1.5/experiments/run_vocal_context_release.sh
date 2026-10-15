#!/usr/bin/env bash
# Minimal causal test: can onset-only trajectory protection preserve lyrics
# while the DiT reads less of the source vocal performance?
#
# Two independent masks are held apart deliberately:
#   1. the lyric alignment plus ONSET_SECONDS=1.0 protects only each line entry;
#   2. the full-vocal F0 mask attenuates source context over the whole vocal.
#
# CONTEXT_STRENGTH is attenuation, so the arm names describe the source-context
# fraction that remains on vocal frames:
#   vctx100 -> strength 0.00 -> retain 100% (control)
#   vctx075 -> strength 0.25 -> retain  75%
#   vctx050 -> strength 0.50 -> retain  50%
#   vctx000 -> strength 1.00 -> retain   0%
set -u

REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
ALIGNMENT="$THESIS_ROOT/experiments/cover_test_kit/02_asr_aligned/chinese_female_120s.json"
FULL_VOCAL_MASK="$THESIS_ROOT/experiments/nonuniform_noise/masks/vocal_mask_stem_3000.npy"
OUT="$THESIS_ROOT/experiments/nonuniform_noise/outputs/vocal_context_release"
PY="$REPO_ROOT/.venv/bin/python"
SEED="${SEED:-20264101}"
STYLE="${STYLE:-jazz}"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACESTEP_VOCAL_MODE=init
export ACESTEP_VOCAL_INIT_STEPS=1
export ACESTEP_VOCAL_ANCHOR_STRENGTH=1.0
export ACESTEP_VOCAL_ONSET_SECONDS=1.0
export ACESTEP_VOCAL_GRANULARITY=line
export ACESTEP_VOCAL_MASK_FADE_SEC=0.25
export ACESTEP_VOCAL_CONTEXT_MASK_NPY="$FULL_VOCAL_MASK"
export ACESTEP_VOCAL_CONTEXT_TARGET=vocal

mkdir -p "$OUT"

run() {
  local name="$1" attenuation="$2" retained="$3"
  local out="$OUT/$name"
  local manifest="$out/manifest_${STYLE}_cns0.00_seed${SEED}.json"
  [ -f "$manifest" ] && { echo "[skip] $name"; return 0; }
  mkdir -p "$out"
  echo "=== $name source_vocal_fraction=$retained ==="
  ACESTEP_VOCAL_CONTEXT_STRENGTH="$attenuation" \
    "$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
    --style "$STYLE" --cover-noise-strength 0.0 --seed "$SEED" --steps 8 \
    --audio-cover-strength 0.8 \
    --vocal-mask "$ALIGNMENT" --vocal-cns 0.375 --mask-fade-seconds 0.25 \
    --source-as-reference --output-dir "$out" > "$OUT/${name}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[FAIL] $name"; else echo "[ok] $name"; fi
  return $rc
}

failures=0
run vctx100 0.00 1.00 || failures=$((failures + 1))
run vctx075 0.25 0.75 || failures=$((failures + 1))
run vctx050 0.50 0.50 || failures=$((failures + 1))
run vctx000 1.00 0.00 || failures=$((failures + 1))

echo "=== vocal-context release sweep finished, failures=$failures ==="
exit $failures
