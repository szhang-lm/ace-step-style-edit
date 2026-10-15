#!/usr/bin/env bash
# The direct "how strongly is the source vocal fed in" knob.
#
# `audio_cover_strength` decides for what fraction of the sampling trajectory the
# source's own acoustic codes are the conditioning, after which the sampler
# switches to an unconditional (silence-latent) context.  At 0.8 the first 80%
# of steps are steered by the source; at 0.5 only half are.  That is a much more
# direct lever on the *singing* than the anchor strength, which only decides how
# hard a vocal frame is written back onto the source path.
#
# Held fixed: jazz, cns=0.0, seed 20264101, 8 steps, anchor strength 0.75,
# mode=init, 1.0 s onset window — i.e. exactly the a075 arm of the anchor sweep
# except for the cover strength.  That makes a075 (acs 0.8) the built-in control.
set -u
REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
ALIGNMENT="$THESIS_ROOT/experiments/cover_test_kit/02_asr_aligned/chinese_female_120s.json"
OUT="$THESIS_ROOT/experiments/nonuniform_noise/outputs/acs_sweep"
PY="$REPO_ROOT/.venv/bin/python"
SEED="${SEED:-20264101}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$OUT"

run() {
  local name="$1" acs="$2"
  local out="$OUT/$name"
  [ -f "$out/manifest_jazz_cns0.00_seed${SEED}.json" ] && { echo "[skip] $name"; return 0; }
  mkdir -p "$out"
  echo "=== $name audio_cover_strength=$acs ==="
  ACESTEP_VOCAL_MODE=init \
  ACESTEP_VOCAL_ANCHOR_STRENGTH=0.75 \
  ACESTEP_VOCAL_ONSET_SECONDS=1.0 \
  ACESTEP_VOCAL_GRANULARITY=line \
  ACESTEP_VOCAL_MASK_FADE_SEC=0.25 \
    "$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
    --style jazz --cover-noise-strength 0.0 --seed "$SEED" --steps 8 \
    --audio-cover-strength "$acs" \
    --vocal-mask "$ALIGNMENT" --vocal-cns 0.6 --mask-fade-seconds 0.25 \
    --source-as-reference --output-dir "$out" > "$OUT/${name}.log" 2>&1
  if [ $? -ne 0 ]; then echo "[FAIL] $name"; else echo "[ok] $name"; fi
}

run acs100 1.0
run acs090 0.9
run acs070 0.7
run acs050 0.5

echo "=== acs sweep finished ==="
