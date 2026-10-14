#!/usr/bin/env bash
# Onset-only protection sweep at the soft_k1 operating point.
#
# Problem under test: protecting a whole lyric line freezes the delivery onto the
# source (corr_vocal 0.36 at soft_k1, 0.82 at k3) and also freezes the
# accompaniment bundled into those frames (high-band energy equals the source).
# Hypothesis: only the *entry* of each line needs to be anchored to keep word
# order and avoid dropped lines; the sustained tail can be re-sung.
set -u
REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
MASK="$THESIS_ROOT/experiments/cover_test_kit/02_asr_aligned/chinese_female_120s.json"
OUT_ROOT="$THESIS_ROOT/experiments/nonuniform_noise/outputs/onset_sweep"
PY="$REPO_ROOT/.venv/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACESTEP_VOCAL_MASK_PATH="$MASK"
export ACESTEP_VOCAL_GRANULARITY=line
export ACESTEP_VOCAL_MASK_FADE_SEC=0.25
export ACESTEP_VOCAL_MODE=init
export ACESTEP_VOCAL_INIT_STEPS=1
export ACESTEP_VOCAL_CNS=0.375     # soft floor, t = 0.625

run() {
  local name="$1" frac="$2" style="${3:-jazz}"
  local out="$OUT_ROOT/$name"
  local manifest="$out/manifest_${style}_cns0.00_seed20264101.json"
  [ -f "$manifest" ] && { echo "[skip] $name"; return 0; }
  mkdir -p "$out"
  echo "=== $name onset_fraction=$frac ==="
  ACESTEP_VOCAL_ONSET_FRACTION="$frac" \
  "$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
    --style "$style" --cover-noise-strength 0.0 --seed 20264101 \
    --vocal-mask "$MASK" --vocal-cns "$ACESTEP_VOCAL_CNS" \
    --source-as-reference --output-dir "$out" \
    > "$OUT_ROOT/../onset_${name}.log" 2>&1
  [ $? -ne 0 ] && echo "[FAIL] $name" || echo "[ok] $name"
}

fail=0
run onset_100 1.0 jazz || fail=$((fail+1))
run onset_050 0.5 jazz || fail=$((fail+1))
run onset_025 0.25 jazz || fail=$((fail+1))
echo "=== onset sweep finished, failures=$fail ==="
exit $fail
