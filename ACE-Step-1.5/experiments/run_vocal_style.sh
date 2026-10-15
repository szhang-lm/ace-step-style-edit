#!/usr/bin/env bash
# Can the VOCAL itself change style while lyrics and melody survive?
#
# Goal this script tests: keep lyrics and melody, change the singing (唱法) to
# jazz.  Two control axes are separated on purpose, because the earlier
# experiments conflated them:
#
#   vocal frames   -- the noise schedule (init/inject) clamps them onto the
#                     source trajectory.  Strong clamping is what produced the
#                     93% lyric coverage, but it also *freezes the performance*,
#                     so the singing can never change.  `mode=none` removes it.
#   accompaniment  -- the source-block gate on `context_latents`.  Turning it
#                     on stops the model from reading the source accompaniment.
#
# Melody is not at risk from either axis: cover mode replaces `src_latents`
# with codes quantized from the source, and measured octave-corrected F0
# correlation is 0.963-0.989 in every arm, including the unprotected one.
#
# Arms (jazz, cns=0.0, seed 20264101, 8 steps, onset window 1.0 s):
#   A_vocal_lock      mode=init  gate off   -- current baseline, lyrics high,
#                                              performance frozen
#   B_vocal_free      mode=none  gate off   -- performance free, lyrics at risk
#   C_free_gate_full  mode=none  gate=1.0   -- free performance + no source
#                                              accompaniment to lean on
#   D_free_gate_half  mode=none  gate=0.5   -- same, gentler
set -u
REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
ALIGNMENT="$THESIS_ROOT/experiments/cover_test_kit/02_asr_aligned/chinese_female_120s.json"
OUT="$THESIS_ROOT/experiments/nonuniform_noise/outputs/vocal_style"
PY="$REPO_ROOT/.venv/bin/python"
SEED="${SEED:-20264101}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$OUT"

run() {
  local name="$1" mode="$2" strength="$3"
  local out="$OUT/$name"
  [ -f "$out/manifest_jazz_cns0.00_seed${SEED}.json" ] && { echo "[skip] $name"; return 0; }
  mkdir -p "$out"
  echo "=== $name mode=$mode context_strength=$strength ==="
  # ACESTEP_VOCAL_CNS must go through --vocal-cns; the runner overwrites the
  # environment variable with that argument's default (0.6) otherwise.
  ACESTEP_VOCAL_MODE="$mode" \
  ACESTEP_VOCAL_CONTEXT_STRENGTH="$strength" \
  ACESTEP_VOCAL_ONSET_SECONDS=1.0 \
  ACESTEP_VOCAL_GRANULARITY=line \
  ACESTEP_VOCAL_MASK_FADE_SEC=0.25 \
    "$PY" "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
    --style jazz --cover-noise-strength 0.0 --seed "$SEED" --steps 8 \
    --vocal-mask "$ALIGNMENT" --vocal-cns 0.6 --mask-fade-seconds 0.25 \
    --source-as-reference --output-dir "$out" > "$OUT/${name}.log" 2>&1
  if [ $? -ne 0 ]; then echo "[FAIL] $name"; else echo "[ok] $name"; fi
}

run A_vocal_lock     init 0.0
run B_vocal_free     none 0.0
run C_free_gate_full none 1.0
run D_free_gate_half none 0.5

echo "=== vocal style sweep finished ==="
