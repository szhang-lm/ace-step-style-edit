#!/usr/bin/env bash
# Feasibility test: use the separated VOCAL STEM as the cover source.
#
# Rationale: a temporal mask cannot separate the vocal from the instruments that
# share its latent frames, and both energy-onset and spectral-flux onset
# detection failed to give a better protection shape.  Feeding a vocals-only
# source attacks the entanglement at the input instead: the model hears the
# vocal with no accompaniment to preserve, so the arrangement must be
# regenerated from the caption.
#
# Timing risk and its mitigation: the stems are sample-aligned to the source
# (+0 ms measured), and this is a SINGLE generation pass, so vocal and
# accompaniment share one timeline.  Only "separate passes, then splice" would
# misalign.  The residual risk is an accent-grid mismatch (the vocal stem's
# onset envelope reads ~176 bpm vs the accompaniment's ~95 bpm).
set -u
REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
STEM="$THESIS_ROOT/experiments/nonuniform_noise/stems"
MASK="$STEM/alignment_trim110.json"
OUT="$THESIS_ROOT/experiments/nonuniform_noise/outputs/stem_test"
PY="$REPO_ROOT/.venv/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACESTEP_VOCAL_MASK_PATH="$MASK"
export ACESTEP_VOCAL_GRANULARITY=line
export ACESTEP_VOCAL_MASK_FADE_SEC=0.25
mkdir -p "$OUT"

run() {
  local name="$1" source="$2" mode="$3" cns="$4"
  echo "=== $name source=$(basename "$source") mode=$mode cns=$cns ==="
  ACESTEP_VOCAL_MODE="$mode" ACESTEP_VOCAL_CNS=0.375 ACESTEP_VOCAL_INIT_STEPS=1 \
  "$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
    --style jazz --cover-noise-strength "$cns" --seed 20264101 \
    --source "$source" --vocal-mask "$MASK" \
    --output-dir "$OUT/$name" > "$OUT/${name}.log" 2>&1
  [ $? -ne 0 ] && echo "[FAIL] $name" || echo "[ok] $name"
}

fail=0
# full mix, no protection: the reference point we already have
run mix_none   "$THESIS_ROOT/experiments/cover_test_kit/sources/chinese_female_120s.wav" none 0.0 || fail=$((fail+1))
# vocals-only stem, no protection
run stem_none  "$STEM/htdemucs/chinese_female_120s/vocals.wav" none 0.0 || fail=$((fail+1))
# vocals-only stem, light onset protection
run stem_onset "$STEM/htdemucs/chinese_female_120s/vocals.wav" init 0.0 || fail=$((fail+1))
echo "=== stem test finished, failures=$fail ==="
exit $fail
