#!/usr/bin/env bash
# Conditioning-side ablation: per-frame gate on the source block of
# `context_latents`.
#
# The cover branch hands the DiT the source's own latents by concatenation:
#   context_latents = cat([src_latents(64ch), chunk_masks], dim=-1)
# so a frame-level multiplier on that block controls how much source
# information the model may READ per frame.  This is a different axis from the
# noise schedule, which controls how far a frame may DRIFT from the source
# trajectory.  Time-domain noise masking provably cannot separate the vocal
# from the accompaniment inside one frame; attenuating the channel can.
#
# Gate form: g = 1 - strength * (1 - mask)
#   strength=0.0 -> exact no-op, bit-identical to stock condition construction
#   strength=1.0 -> vocal frames read the full source block, accompaniment
#                   frames read none of it
#
# The arms below pair the gate with the LIGHTEST noise protection (init, 1
# step) on purpose: the noise schedule is then nearly a no-op, so any change
# in the output is attributable to the conditioning gate rather than to the
# renoising.  The F0/stem mask is used because it is the shape that leaves the
# most accompaniment frames ungated (mask mean 0.4873), which maximises the
# contrast the gate can create.
set -u
REPO_ROOT="/Users/kkszhang/project/毕设/ACE-Step-1.5"
THESIS_ROOT="/Users/kkszhang/project/毕设"
STEM_MASK="$THESIS_ROOT/experiments/nonuniform_noise/masks/vocal_mask_stem_3000.npy"
# The loader short-circuits to "no schedule" unless a mask path *or* a
# precomputed .npy is configured, and the runner only exports
# ACESTEP_VOCAL_MASK_PATH when --vocal-mask is passed.  Pass the alignment as
# that path so the loader is entered; ACESTEP_VOCAL_MASK_NPY wins from there,
# so the alignment itself is never read.
ALIGNMENT="$THESIS_ROOT/experiments/cover_test_kit/02_asr_aligned/chinese_female_120s.json"
OUT="$THESIS_ROOT/experiments/nonuniform_noise/outputs/context_gate"
PY="$REPO_ROOT/.venv/bin/python"
SEED="${SEED:-20264101}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACESTEP_VOCAL_MASK_NPY="$STEM_MASK"
export ACESTEP_VOCAL_GRANULARITY=line
export ACESTEP_VOCAL_MASK_FADE_SEC=0.25
export ACESTEP_VOCAL_MODE=init
export ACESTEP_VOCAL_INIT_STEPS=1
export ACESTEP_VOCAL_CNS=0.375

mkdir -p "$OUT"

run() {
  local name="$1" strength="$2"
  local out="$OUT/$name"
  [ -f "$out/manifest_jazz_cns0.00_seed${SEED}.json" ] && { echo "[skip] $name"; return 0; }
  mkdir -p "$out"
  echo "=== $name context_strength=$strength ==="
  # --vocal-cns must be passed explicitly: the runner overrides
  # ACESTEP_VOCAL_CNS with this argument's default (0.6), so exporting it from
  # the environment is silently ignored whenever --vocal-mask is given.
  ACESTEP_VOCAL_CONTEXT_STRENGTH="$strength" "$PY" \
    "$REPO_ROOT/experiments/run_chinese_female_cover_mps.py" \
    --style jazz --cover-noise-strength 0.0 --seed "$SEED" --steps 8 \
    --vocal-mask "$ALIGNMENT" --vocal-cns 0.375 --mask-fade-seconds 0.25 \
    --source-as-reference --output-dir "$out" > "$OUT/${name}.log" 2>&1
  if [ $? -ne 0 ]; then echo "[FAIL] $name"; else echo "[ok] $name"; fi
}

# g0_off is the control: identical noise protection, gate disabled.  It is the
# same configuration as the main ablation's k1 arm except for the mask source,
# so it doubles as a check that the stem mask alone changes little.
run g0_off   0.0
run g1_full  1.0
run g5_half  0.5

echo "=== context gate sweep finished ==="
