#!/usr/bin/env bash
# Multi-seed replication of the cns=0.0 ablation.
#
# The sampler changes run to run even at a fixed seed (MPS kernels are not
# deterministic and no deterministic-algorithms mode is set), and same-seed CLAP
# distance varied by 0.26 -- larger than the effect under test.  Lyric coverage
# was stable to +-2 points, so it is the primary endpoint, but every arm still
# needs several seeds before any claim is made.
set -u
REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
MASK="$THESIS_ROOT/experiments/cover_test_kit/02_asr_aligned/chinese_female_120s.json"
OUT_ROOT="$THESIS_ROOT/experiments/nonuniform_noise/outputs"

PY="$REPO_ROOT/.venv/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACESTEP_VOCAL_MASK_PATH="$MASK"
export ACESTEP_VOCAL_GRANULARITY=line
export ACESTEP_VOCAL_MASK_FADE_SEC=0.25
export ACESTEP_VOCAL_CNS=0.625

run_arm() {
  local seed="$1" arm="$2" mode="$3" k="$4" style="$5"
  local out="$OUT_ROOT/ablation_s${seed}/$arm"
  local manifest="$out/manifest_${style}_cns0.00_seed${seed}.json"
  [ -f "$manifest" ] && { echo "[skip] s$seed $arm/$style"; return 0; }
  mkdir -p "$out"
  echo "=== seed=$seed $arm mode=$mode k=$k style=$style ==="
  ACESTEP_VOCAL_MODE="$mode" ACESTEP_VOCAL_INIT_STEPS="$k" \
  "$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
    --style "$style" --cover-noise-strength 0.0 --seed "$seed" \
    --vocal-mask "$MASK" --vocal-cns "$ACESTEP_VOCAL_CNS" \
    --source-as-reference --output-dir "$out" \
    > "$out/../${arm}_${style}.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL] s$seed $arm/$style rc=$rc"; return $rc; }
  echo "[ok] s$seed $arm/$style"
}

fail=0
for seed in 20264101 20264102 20264103; do
  for style in jazz rock; do
    run_arm "$seed" k0 none 1 "$style" || fail=$((fail+1))
    run_arm "$seed" k1 init 1 "$style" || fail=$((fail+1))
    run_arm "$seed" k3 init 3 "$style" || fail=$((fail+1))
    run_arm "$seed" k8 init 8 "$style" || fail=$((fail+1))
  done
done
echo "=== seeds finished, failures=$fail ==="
exit $fail
