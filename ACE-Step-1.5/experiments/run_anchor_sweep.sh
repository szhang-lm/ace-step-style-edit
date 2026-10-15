#!/usr/bin/env bash
# How much of the source vocal should be fed back in?
#
# The earlier arms only had the two extremes: `inject` wrote the vocal frame
# entirely onto the source path (anchor 1.0) and `none` left it entirely alone
# (anchor 0.0).  Neither can express "let the source vocal anchor the delivery
# without dictating it", which is the middle of that range.
#
# ACESTEP_VOCAL_ANCHOR_STRENGTH makes the anchor continuous: a vocal frame keeps
# (1 - s) of its own generated value and takes s of the source path.  This
# sweep walks s from 1.0 (the old hard clamp) down to 0.25, holding everything
# else fixed, so the effect is attributable to the anchor alone.
#
# Held fixed: jazz, cns=0.0, seed 20264101, 8 steps, mode=init (anchor applied
# at the first step), 1.0 s onset window on the alignment mask, max_onset_peaks
# default.  The onset window keeps only the first 1.0 s of each lyric line
# (~8% of frames), i.e. the lyric onsets anchor the words while the rest of the
# performance is left to the model.
set -u
REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
ALIGNMENT="$THESIS_ROOT/experiments/cover_test_kit/02_asr_aligned/chinese_female_120s.json"
OUT="$THESIS_ROOT/experiments/nonuniform_noise/outputs/anchor_sweep"
PY="$REPO_ROOT/.venv/bin/python"
SEED="${SEED:-20264101}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$OUT"

run() {
  local name="$1" anchor="$2"
  local out="$OUT/$name"
  [ -f "$out/manifest_jazz_cns0.00_seed${SEED}.json" ] && { echo "[skip] $name"; return 0; }
  mkdir -p "$out"
  echo "=== $name anchor_strength=$anchor ==="
  ACESTEP_VOCAL_MODE=init \
  ACESTEP_VOCAL_ANCHOR_STRENGTH="$anchor" \
  ACESTEP_VOCAL_ONSET_SECONDS=1.0 \
  ACESTEP_VOCAL_GRANULARITY=line \
  ACESTEP_VOCAL_MASK_FADE_SEC=0.25 \
    "$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
    --style jazz --cover-noise-strength 0.0 --seed "$SEED" --steps 8 \
    --vocal-mask "$ALIGNMENT" --vocal-cns 0.6 --mask-fade-seconds 0.25 \
    --source-as-reference --output-dir "$out" > "$OUT/${name}.log" 2>&1
  if [ $? -ne 0 ]; then echo "[FAIL] $name"; else echo "[ok] $name"; fi
}

run a100 1.0
run a075 0.75
run a050 0.5
run a025 0.25

echo "=== anchor sweep finished ==="
