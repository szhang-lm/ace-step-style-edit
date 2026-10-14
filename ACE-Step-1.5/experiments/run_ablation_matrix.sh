#!/usr/bin/env bash
# Ablation matrix for vocal-aware non-uniform cover renoising.
#
# The style knob is fixed at cover_noise_strength=0.0 in EVERY arm.  Listening
# plus CLAP distance established that only this setting produces a large style
# change (0.48 vs 0.03 at cns=0.6), so it is the only setting where protecting
# the vocals has anything to trade off against.  With global t = 1.0 the
# accompaniment starts from pure noise, while the vocal frames are held at the
# t=0.375 vocal floor (ACESTEP_VOCAL_CNS=0.625) -- that split IS the non-uniform
# schedule, and it is exactly the "init" mechanism.
#
# The arms therefore differ only in how long the vocal frames are held there:
#
#   k0  no protection at all           (baseline: style changes, lyrics suffer)
#   k1  non-uniform start, first step only
#   k3  held for three steps
#   k8  held for every step in the loop (upper bound / over-protection probe)
#
# Every arm passes --source-as-reference so the acoustic reference is the source
# song, matching the archived baseline runs.
#
# Run:  bash experiments/run_ablation_matrix.sh [style ...]
set -u

REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
MASK="$THESIS_ROOT/experiments/cover_test_kit/02_asr_aligned/chinese_female_120s.json"
OUT_ROOT="$THESIS_ROOT/experiments/nonuniform_noise/outputs/ablation"

STYLES=("$@")
if [ ${#STYLES[@]} -eq 0 ]; then
  STYLES=(jazz rock)
fi

PY="$REPO_ROOT/.venv/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACESTEP_VOCAL_MASK_PATH="$MASK"
export ACESTEP_VOCAL_GRANULARITY=line
export ACESTEP_VOCAL_MASK_FADE_SEC=0.25
export ACESTEP_VOCAL_CNS=0.625   # vocal floor t = 0.375
CNS=0.0

run_arm() {
  local name="$1" mode="$2" init_steps="$3" style="$4"
  local out="$OUT_ROOT/$name"
  local manifest="$out/manifest_${style}_cns0.00_seed20264000.json"
  if [ -f "$manifest" ]; then
    echo "[skip] $name/$style already done"
    return 0
  fi
  echo "=== $name | style=$style mode=$mode init_steps=$init_steps ==="
  ACESTEP_VOCAL_MODE="$mode" \
  ACESTEP_VOCAL_INIT_STEPS="$init_steps" \
  "$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
    --style "$style" \
    --cover-noise-strength "$CNS" \
    --vocal-mask "$MASK" \
    --vocal-cns "$ACESTEP_VOCAL_CNS" \
    --source-as-reference \
    --output-dir "$out" \
    > "$OUT_ROOT/${name}_${style}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] $name/$style rc=$rc (see $OUT_ROOT/${name}_${style}.log)"
  else
    echo "[ok] $name/$style"
  fi
  return $rc
}

mkdir -p "$OUT_ROOT"
failures=0
for style in "${STYLES[@]}"; do
  run_arm k0 none   1 "$style" || failures=$((failures + 1))
  run_arm k1 init   1 "$style" || failures=$((failures + 1))
  run_arm k3 init   3 "$style" || failures=$((failures + 1))
  run_arm k8 init   8 "$style" || failures=$((failures + 1))
done

echo "=== matrix finished, failures=$failures ==="
exit $failures
