#!/usr/bin/env bash
# Protection-strength sweep at cns=0.0.
#
# k8 showed the protected arms win on lyric coverage partly by copying the source
# vocal (corr_vocal 0.93, NMSE -8.4 dB).  Coverage alone cannot tell "the model
# sang it" from "we wrote the source back", so every arm is scored on BOTH:
#   - corr_vocal  (source reuse; want 0.2-0.4, not >0.8)
#   - corr_bg     (must stay low: the accompaniment really did change style)
#   - ASR coverage (want >= 88%)
# k  = number of steps the vocal frames are anchored
# cns_vocal = vocal floor; 1-t = cns, so 0.625 -> t=0.375 (hard), 0.375 -> t=0.625 (soft)
set -u
REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
MASK="$THESIS_ROOT/experiments/cover_test_kit/02_asr_aligned/chinese_female_120s.json"
OUT_ROOT="$THESIS_ROOT/experiments/nonuniform_noise/outputs/sweep"
PY="$REPO_ROOT/.venv/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACESTEP_VOCAL_MASK_PATH="$MASK"
export ACESTEP_VOCAL_GRANULARITY=line
export ACESTEP_VOCAL_MASK_FADE_SEC=0.25

run() {
  local name="$1" mode="$2" k="$3" vcns="$4" style="$5"
  local out="$OUT_ROOT/$name"
  local manifest="$out/manifest_${style}_cns0.00_seed20264101.json"
  [ -f "$manifest" ] && { echo "[skip] $name"; return 0; }
  mkdir -p "$out"
  echo "=== $name mode=$mode k=$k vocal_cns=$vcns ==="
  ACESTEP_VOCAL_MODE="$mode" ACESTEP_VOCAL_INIT_STEPS="$k" ACESTEP_VOCAL_CNS="$vcns" \
  "$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
    --style "$style" --cover-noise-strength 0.0 --seed 20264101 \
    --vocal-mask "$MASK" --vocal-cns "$vcns" \
    --source-as-reference --output-dir "$out" \
    > "$OUT_ROOT/../sweep_${name}.log" 2>&1
  [ $? -ne 0 ] && echo "[FAIL] $name" || echo "[ok] $name"
}

fail=0
for style in jazz; do
  run soft_k1  init 1 0.375 "$style" || fail=$((fail+1))   # only init, t=0.625
  run soft_k3  init 3 0.375 "$style" || fail=$((fail+1))   # 3 steps, t=0.625
  run mid_k2   init 2 0.5   "$style" || fail=$((fail+1))   # 2 steps, t=0.5
  run mid_k1   init 1 0.5   "$style" || fail=$((fail+1))   # only init, t=0.5
done
echo "=== sweep finished, failures=$fail ==="
exit $fail
