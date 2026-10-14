#!/usr/bin/env bash
# Does a finer sampling trajectory buy anything?
#
# At 8 steps the noise floor can only sit on 6 discrete points and, after the
# vocal floor (t=0.375) is reached, the model has only 3 steps left to deviate
# from the source.  At 20 steps the floor is finer AND there are ~11 free steps
# after release, which is the "smoother trajectory" the method wants.
#
# Arms (all cns=0.0, jazz, seed 20264101, onset window 1.0s, vocal floor t=0.375):
#   s8_k1 / s20_k1    1 protected step
#   s8_k3 / s20_k3    3 protected steps
#   s20_k11           protected until the floor is reached at 20 steps
set -u
REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
MASK="$THESIS_ROOT/experiments/cover_test_kit/02_asr_aligned/chinese_female_120s.json"
OUT="$THESIS_ROOT/experiments/nonuniform_noise/outputs/steps_test"
PY="$REPO_ROOT/.venv/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACESTEP_VOCAL_MASK_PATH="$MASK"
export ACESTEP_VOCAL_GRANULARITY=line
export ACESTEP_VOCAL_MASK_FADE_SEC=0.25
export ACESTEP_VOCAL_MODE=init
export ACESTEP_VOCAL_CNS=0.375
export ACESTEP_VOCAL_ONSET_SECONDS=1.0
mkdir -p "$OUT"

run() {
  local name="$1" steps="$2" k="$3"
  local out="$OUT/$name"
  [ -f "$out/manifest_jazz_cns0.00_seed20264101.json" ] && { echo "[skip] $name"; return 0; }
  mkdir -p "$out"
  echo "=== $name steps=$steps init_steps=$k ==="
  ACESTEP_VOCAL_INIT_STEPS="$k" "$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
    --style jazz --cover-noise-strength 0.0 --seed 20264101 --steps "$steps" \
    --source-as-reference --output-dir "$out" > "$OUT/${name}.log" 2>&1
  [ $? -ne 0 ] && echo "[FAIL] $name" || echo "[ok] $name"
}

fail=0
run s8_k1  8  1 || fail=$((fail+1))
run s20_k1 20 1 || fail=$((fail+1))
run s20_k3 20 3 || fail=$((fail+1))
run s20_k11 20 11 || fail=$((fail+1))
echo "=== steps test finished, failures=$fail ==="
exit $fail
