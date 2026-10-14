#!/usr/bin/env bash
# Fixed-duration onset-window sweep at the soft operating point.
#
# The earlier "onset25" result actually protected the first 1.0s of each lyric
# line (a 1.0s cap inside the mask builder), so the operative knob is the window
# LENGTH, not a fraction.  This sweep calibrates it: longer window = more lyric
# anchoring but more source delivery frozen onto the output.
set -u
REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
MASK="$THESIS_ROOT/experiments/cover_test_kit/02_asr_aligned/chinese_female_120s.json"
OUT_ROOT="$THESIS_ROOT/experiments/nonuniform_noise/outputs/window_sweep"
PY="$REPO_ROOT/.venv/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACESTEP_VOCAL_MASK_PATH="$MASK"
export ACESTEP_VOCAL_GRANULARITY=line
export ACESTEP_VOCAL_MASK_FADE_SEC=0.25
export ACESTEP_VOCAL_MODE=init
export ACESTEP_VOCAL_INIT_STEPS=1
export ACESTEP_VOCAL_CNS=0.375      # soft floor t=0.625

run() {
  local name="$1" sec="$2"
  local out="$OUT_ROOT/$name"
  local manifest="$out/manifest_jazz_cns0.00_seed20264101.json"
  [ -f "$manifest" ] && { echo "[skip] $name"; return 0; }
  mkdir -p "$out"
  echo "=== $name onset_seconds=$sec ==="
  if [ "$sec" = "FULL" ]; then
    env -u ACESTEP_VOCAL_ONSET_SECONDS -u ACESTEP_VOCAL_ONSET_FRACTION \
    "$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
      --style jazz --cover-noise-strength 0.0 --seed 20264101 \
      --vocal-mask "$MASK" --vocal-cns "$ACESTEP_VOCAL_CNS" \
      --source-as-reference --output-dir "$out" \
      > "$OUT_ROOT/../window_${name}.log" 2>&1
  else
    ACESTEP_VOCAL_ONSET_SECONDS="$sec" \
    "$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
      --style jazz --cover-noise-strength 0.0 --seed 20264101 \
      --vocal-mask "$MASK" --vocal-cns "$ACESTEP_VOCAL_CNS" \
      --source-as-reference --output-dir "$out" \
      > "$OUT_ROOT/../window_${name}.log" 2>&1
  fi
  [ $? -ne 0 ] && echo "[FAIL] $name" || echo "[ok] $name"
}

fail=0
run win_0p4 0.4 || fail=$((fail+1))
run win_1p0 1.0 || fail=$((fail+1))
run win_1p5 1.5 || fail=$((fail+1))
echo "=== window sweep finished, failures=$fail ==="
exit $fail
