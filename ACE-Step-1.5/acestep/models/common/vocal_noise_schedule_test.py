"""Tests for the experimental vocal-aware noise schedule."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from acestep.models.common.vocal_noise_schedule import (
    ANCHOR_STRENGTH_ENV,
    CONTEXT_MASK_NPY_ENV,
    CONTEXT_STRENGTH_ENV,
    CONTEXT_TARGET_ENV,
    INIT_STEPS_ENV,
    MASK_FADE_ENV,
    MASK_PATH_ENV,
    MODE_ENV,
    VOCAL_CNS_ENV,
    VocalProtectionSpec,
    anchor_strength_from_env,
    apply_source_context_gate,
    build_vocal_activity_mask,
    context_gate_strength_from_env,
    context_gate_target_from_env,
    extract_intervals,
    inject_source_path,
    load_context_gate_mask_from_env,
    load_vocal_schedule_from_env,
    nearest_timestep,
    prepare_nonuniform_start,
    resolve_mode,
    source_path_at,
)


THESIS_ROOT = Path(__file__).resolve().parents[4]
ALIGNMENT_PATH = (
    THESIS_ROOT
    / "experiments"
    / "cover_test_kit"
    / "02_asr_aligned"
    / "chinese_female_120s.json"
)

# The reference alignment's first token is stretched over 15.8s by the forced
# aligner, so token granularity is the *unreliable* option and line granularity
# is the default.  Both behaviours are pinned below.
REFERENCE_VOCAL_INTERVALS = [
    (0.64, 24.96),
    (25.2, 35.52),
    (38.56, 46.8),
    (46.8, 52.16),
    (52.24, 62.96),
    (65.04, 74.4),
    (75.76, 84.64),
    (87.68, 103.36),
]


def _spec(mask, vocal_timestep, mode, init_steps=1) -> VocalProtectionSpec:
    return VocalProtectionSpec(
        mask=mask,
        vocal_timestep=vocal_timestep,
        mode=mode,
        init_steps=init_steps,
        granularity="line",
        mask_path=Path("/dev/null"),
        vocal_cns=1.0 - vocal_timestep,
        fade_seconds=0.25,
    )


class VocalNoiseScheduleTest(unittest.TestCase):
    """Cover mask construction and non-uniform renoising behavior."""

    def test_repairs_anomalously_long_first_token(self) -> None:
        """A bad 0.64-16.4 alignment must not protect the entire intro."""
        alignment = {"tokens": [{"text": "让", "start": 0.64, "end": 16.4}]}
        mask = build_vocal_activity_mask(
            alignment, 500, fade_seconds=0.0, granularity="token"
        )
        self.assertEqual(mask.shape, (1, 500, 1))
        self.assertEqual(mask[0, 25, 0].item(), 0.0)
        self.assertEqual(mask[0, 390, 0].item(), 1.0)

    def test_soft_mask_blends_generated_and_source_paths(self) -> None:
        """Mask values zero, half, and one produce the expected interpolation."""
        xt = torch.tensor([[[10.0], [10.0], [10.0]]])
        src = torch.zeros_like(xt)
        noise = torch.full_like(xt, 2.0)
        mask = torch.tensor([[[0.0], [0.5], [1.0]]])
        result = inject_source_path(xt, src, noise, mask, timestep=0.5)
        torch.testing.assert_close(result, torch.tensor([[[10.0], [5.5], [1.0]]]))

    def test_line_intervals_match_the_source_alignment(self) -> None:
        """Line-level extraction reproduces the alignment's own line bounds."""
        alignment = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))
        intervals = extract_intervals(alignment, granularity="line")
        self.assertEqual(len(intervals), len(REFERENCE_VOCAL_INTERVALS))
        for (got_s, got_e), (want_s, want_e) in zip(intervals, REFERENCE_VOCAL_INTERVALS):
            self.assertLessEqual(abs(got_s - want_s), 0.05, f"start {got_s} vs {want_s}")
            self.assertLessEqual(abs(got_e - want_e), 0.05, f"end {got_e} vs {want_e}")

    def test_line_mask_keeps_interludes_unprotected(self) -> None:
        """Gaps between lines must stay at zero while vocal regions stay at one."""
        alignment = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))
        mask = build_vocal_activity_mask(alignment, 3000, fade_seconds=0.0)
        self.assertGreaterEqual(mask.mean().item(), 0.70)
        self.assertLessEqual(mask.mean().item(), 0.82)
        for second in (37.0, 36.2, 85.5, 86.5):
            self.assertEqual(mask[0, int(second * 25), 0].item(), 0.0)
        self.assertEqual(mask[0, int(10.0 * 25), 0].item(), 1.0)

    def test_token_granularity_under_covers_the_intro(self) -> None:
        """Token granularity leaves the warped intro bare; line granularity does not."""
        alignment = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))
        token_mask = build_vocal_activity_mask(
            alignment, 3000, fade_seconds=0.0, granularity="token"
        )
        line_mask = build_vocal_activity_mask(
            alignment, 3000, fade_seconds=0.0, granularity="line"
        )
        self.assertEqual(token_mask[0, int(5.0 * 25), 0].item(), 0.0)
        self.assertEqual(line_mask[0, int(5.0 * 25), 0].item(), 1.0)
        self.assertLess(token_mask.mean().item(), line_mask.mean().item())

    def test_nearest_timestep_snaps_to_schedule(self) -> None:
        """Requested noise levels snap onto the nearest sampler timestep."""
        schedule = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
        self.assertEqual(nearest_timestep(schedule, 0.4), 0.375)
        self.assertEqual(nearest_timestep(schedule, 0.95), 1.0)
        self.assertEqual(nearest_timestep(schedule, 0.13), 0.125)

    def test_none_mode_reproduces_the_stock_global_renoise(self) -> None:
        """mode='none' must be an exact no-op relative to the stock sampler."""
        src = torch.randn(1, 8, 4)
        noise = torch.randn(1, 8, 4)
        mask = torch.ones(1, 8, 1)
        spec = _spec(mask, vocal_timestep=0.375, mode="none")
        got = prepare_nonuniform_start(src, noise, spec, global_timestep=0.75)
        want = 0.75 * noise + 0.25 * src
        torch.testing.assert_close(got, want)

    def test_init_mode_lowers_only_vocal_frames(self) -> None:
        """mode='init' puts vocal frames at vocal_t and accompaniment at global_t."""
        src = torch.randn(1, 4, 1)
        noise = torch.randn(1, 4, 1)
        mask = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
        spec = _spec(mask, vocal_timestep=0.25, mode="init")
        got = prepare_nonuniform_start(src, noise, spec, global_timestep=0.75)
        torch.testing.assert_close(got[:, :2], source_path_at(src, noise, 0.25)[:, :2])
        torch.testing.assert_close(got[:, 2:], source_path_at(src, noise, 0.75)[:, 2:])

    def test_init_and_inject_write_the_same_latent(self) -> None:
        """The two mechanisms write the same quantity when they fire together.

        This is the equivalence the ablation rests on: 'let vocal frames enter
        later at a lower noise level' and 'clamp them back onto the source path
        while above that level' are the same operation on the latent.
        """
        src = torch.randn(1, 6, 1)
        noise = torch.randn(1, 6, 1)
        mask = torch.tensor([[[1.0], [1.0], [0.5], [0.0], [1.0], [0.0]]])
        init_spec = _spec(mask, vocal_timestep=0.375, mode="init")
        both_spec = _spec(mask, vocal_timestep=0.375, mode="both")

        started = prepare_nonuniform_start(src, noise, init_spec, global_timestep=0.75)
        clamped = prepare_nonuniform_start(src, noise, both_spec, global_timestep=0.75)
        torch.testing.assert_close(started, clamped)

        # Fully protected frames sit exactly on the source path at the vocal floor.
        full_idx = torch.nonzero(mask.squeeze(-1)[0] == 1.0).flatten()
        half_idx = torch.nonzero(mask.squeeze(-1)[0] == 0.5).flatten()
        zero_idx = torch.nonzero(mask.squeeze(-1)[0] == 0.0).flatten()
        self.assertGreater(full_idx.numel(), 0)
        self.assertGreater(half_idx.numel(), 0)
        self.assertGreater(zero_idx.numel(), 0)
        source_at_floor = source_path_at(src, noise, 0.375)
        source_at_global = source_path_at(src, noise, 0.75)
        torch.testing.assert_close(started[:, full_idx, :], source_at_floor[:, full_idx, :])
        torch.testing.assert_close(started[:, zero_idx, :], source_at_global[:, zero_idx, :])
        # Fade frames land halfway between the global and vocal source paths.
        torch.testing.assert_close(
            started[:, half_idx, :],
            0.5 * (source_at_global[:, half_idx, :] + source_at_floor[:, half_idx, :]),
        )

        # Re-clamping keeps protected frames on the source path (up to float drift)
        # and never pulls unprotected frames towards the source.
        reinject = inject_source_path(started, src, noise, mask, 0.375)
        torch.testing.assert_close(
            reinject[:, full_idx, :], source_at_floor[:, full_idx, :], atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(reinject[:, zero_idx, :], started[:, zero_idx, :])
        self.assertGreater(
            (reinject[:, half_idx, :] - source_at_floor[:, half_idx, :]).abs().max().item(),
            0.0,
        )

    def test_protects_step_matches_the_ablation_arms(self) -> None:
        """Only the intended steps fire in each mode, never below the vocal floor.

        The sampling loop's last iteration predicts x0 instead of taking a step,
        so it must never clamp -- clamping the clean sample would drag the vocals
        back onto the source and undo the style change entirely.  Hence the
        expected sets below exclude the final step.
        """
        schedule = [0.75, 0.625, 0.5, 0.375, 0.25]
        floor = 0.375
        arms = {
            "init": [0, 1, 2],
            "inject": [1, 2, 3],
            "both": [0, 1, 2, 3],
            "none": [],
        }
        for mode, expected in arms.items():
            spec = _spec(torch.ones(1, 2, 1), vocal_timestep=floor, mode=mode, init_steps=3)
            protected = [
                idx
                for idx, t_after in enumerate(schedule[:-1])
                if spec.protects_step(idx) and t_after >= floor
            ]
            self.assertEqual(protected, expected, f"mode={mode}")

    def test_init_mode_fires_only_on_its_configured_prefix(self) -> None:
        """``init_steps`` bounds the init arm to the first N steps."""
        spec = _spec(torch.ones(1, 2, 1), vocal_timestep=0.375, mode="init", init_steps=2)
        self.assertEqual([idx for idx in range(6) if spec.protects_step(idx)], [0, 1])

    def test_resolve_mode_handles_empty_and_rejects_typos(self) -> None:
        """Unset values mean 'none'; typos are hard errors."""
        self.assertEqual(resolve_mode(None), "none")
        self.assertEqual(resolve_mode(""), "none")
        self.assertEqual(resolve_mode("BOTH"), "both")
        with self.assertRaises(ValueError):
            resolve_mode("initial")

    def test_env_loader_rejects_vocal_noise_above_global(self) -> None:
        """A vocal floor noisier than the global start is a configuration error."""
        alignment = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alignment.json"
            path.write_text(json.dumps(alignment), encoding="utf-8")
            saved = {
                key: os.environ.get(key)
                for key in (MASK_PATH_ENV, VOCAL_CNS_ENV, MODE_ENV, MASK_FADE_ENV, INIT_STEPS_ENV)
            }
            try:
                os.environ[MASK_PATH_ENV] = str(path)
                os.environ[MODE_ENV] = "both"
                os.environ[VOCAL_CNS_ENV] = "0.1"  # -> vocal_t 0.9 > global 0.75
                src = torch.zeros(1, 3000, 32)
                with self.assertRaises(ValueError):
                    load_vocal_schedule_from_env(
                        src,
                        valid_timesteps=[1.0, 0.875, 0.75, 0.625],
                        global_timestep=0.75,
                    )
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_env_loader_is_off_without_a_mask_path(self) -> None:
        """No configured mask means the sampler keeps its stock behaviour."""
        saved = os.environ.pop(MASK_PATH_ENV, None)
        try:
            self.assertIsNone(load_vocal_schedule_from_env(torch.zeros(1, 10, 4)))
        finally:
            if saved is not None:
                os.environ[MASK_PATH_ENV] = saved


class SourceContextGateTest(unittest.TestCase):
    """The conditioning-side gate that attenuates the source block per frame."""

    @staticmethod
    def _mask(values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32).view(1, -1, 1)

    def test_zero_strength_is_an_exact_no_op(self) -> None:
        """strength=0 must reproduce stock condition construction bit for bit."""
        spec = _spec(self._mask([[1.0, 0.0, 1.0]]), 0.5, "both")
        gate = spec.context_gate(0.0)
        self.assertTrue(torch.equal(gate, torch.ones_like(spec.mask)))

        context = torch.randn(1, 3, 8)
        self.assertTrue(
            torch.equal(apply_source_context_gate(context, gate, 6), context)
        )

    def test_full_strength_hands_the_gate_the_mask_itself(self) -> None:
        """strength=1 protects exactly the frames the noise mask protects."""
        mask = self._mask([[1.0, 0.0, 0.5]])
        spec = _spec(mask, 0.5, "both")
        self.assertTrue(torch.equal(spec.context_gate(1.0), mask))

    def test_intermediate_strength_blends_towards_the_mask(self) -> None:
        """Half strength keeps half of the accompaniment's source block."""
        mask = self._mask([[1.0, 0.0]])
        spec = _spec(mask, 0.5, "both")
        gate = spec.context_gate(0.5)
        self.assertAlmostEqual(gate[0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(gate[0, 1, 0].item(), 0.5)

    def test_vocal_target_attenuates_the_inverse_region(self) -> None:
        """The vocal target frees vocal frames and preserves accompaniment."""
        mask = self._mask([[1.0, 0.0, 0.5]])
        spec = _spec(mask, 0.5, "init")
        gate = spec.context_gate(1.0, target="vocal")
        torch.testing.assert_close(gate, self._mask([[0.0, 1.0, 0.5]]))

    def test_context_mask_can_differ_from_noise_mask(self) -> None:
        """A full-vocal context mask need not reuse the onset protection mask."""
        onset_mask = self._mask([[1.0, 0.0, 0.0]])
        full_vocal_mask = self._mask([[1.0, 1.0, 0.0]])
        spec = _spec(onset_mask, 0.5, "init")
        gate = spec.context_gate(1.0, mask=full_vocal_mask, target="vocal")
        torch.testing.assert_close(gate, self._mask([[0.0, 0.0, 1.0]]))
        self.assertTrue(torch.equal(spec.mask, onset_mask))

    def test_out_of_range_strengths_are_rejected(self) -> None:
        spec = _spec(self._mask([[1.0, 0.0]]), 0.5, "both")
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                spec.context_gate(bad)

    def test_unknown_target_is_rejected(self) -> None:
        spec = _spec(self._mask([[1.0, 0.0]]), 0.5, "init")
        with self.assertRaises(ValueError):
            spec.context_gate(0.5, target="everything")

    def test_gate_scales_only_the_source_block(self) -> None:
        """The chunk-mask channels mark region validity and stay untouched.

        The channel ratio here is 2 source + 1 validity, i.e. deliberately
        *not* 1:1: the real layout is 128 source channels plus one mask
        channel, so any implementation that infers the boundary by halving
        the channel count must fail this test.
        """
        source = torch.stack([torch.zeros(2), torch.full((2,), 10.0)]).view(1, 2, 2)
        validity = torch.ones(1, 2, 1)
        context = torch.cat([source, validity], dim=-1)
        gate = self._mask([[1.0, 0.0]])
        gated = apply_source_context_gate(context, gate, 2)

        self.assertEqual(gated.shape, (1, 2, 3))
        torch.testing.assert_close(gated[..., 2:], validity)
        torch.testing.assert_close(gated[0, 0, :2], source[0, 0])
        torch.testing.assert_close(gated[0, 1, :2], torch.zeros(2))

    def test_split_width_must_be_consistent(self) -> None:
        context = torch.randn(1, 4, 6)
        for bad in (0, 6, 7, -1):
            with self.assertRaises(ValueError):
                apply_source_context_gate(context, self._mask([[1.0] * 4]), bad)

    def test_none_gate_leaves_context_untouched(self) -> None:
        context = torch.randn(1, 5, 6)
        self.assertTrue(
            torch.equal(apply_source_context_gate(context, None, 5), context)
        )

    def test_env_strength_defaults_to_off_and_validates(self) -> None:
        saved = os.environ.pop(CONTEXT_STRENGTH_ENV, None)
        try:
            self.assertEqual(context_gate_strength_from_env(), 0.0)
            os.environ[CONTEXT_STRENGTH_ENV] = "0.75"
            self.assertEqual(context_gate_strength_from_env(), 0.75)
            os.environ[CONTEXT_STRENGTH_ENV] = "1.5"
            with self.assertRaises(ValueError):
                context_gate_strength_from_env()
        finally:
            if saved is None:
                os.environ.pop(CONTEXT_STRENGTH_ENV, None)
            else:
                os.environ[CONTEXT_STRENGTH_ENV] = saved

    def test_env_target_defaults_to_accompaniment_and_validates(self) -> None:
        saved = os.environ.pop(CONTEXT_TARGET_ENV, None)
        try:
            self.assertEqual(context_gate_target_from_env(), "accompaniment")
            os.environ[CONTEXT_TARGET_ENV] = "vocal"
            self.assertEqual(context_gate_target_from_env(), "vocal")
            os.environ[CONTEXT_TARGET_ENV] = "invalid"
            with self.assertRaises(ValueError):
                context_gate_target_from_env()
        finally:
            if saved is None:
                os.environ.pop(CONTEXT_TARGET_ENV, None)
            else:
                os.environ[CONTEXT_TARGET_ENV] = saved

    def test_context_mask_loader_uses_an_independent_npy(self) -> None:
        fallback = self._mask([[1.0, 0.0, 0.0]])
        src = torch.zeros(2, 3, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full_vocal.npy"
            np.save(path, np.array([0.0, 0.5, 1.0], dtype=np.float32))
            saved = os.environ.get(CONTEXT_MASK_NPY_ENV)
            try:
                os.environ[CONTEXT_MASK_NPY_ENV] = str(path)
                loaded = load_context_gate_mask_from_env(src, fallback)
                self.assertEqual(loaded.shape, (2, 3, 1))
                torch.testing.assert_close(loaded[0], self._mask([[0.0, 0.5, 1.0]])[0])
                self.assertTrue(torch.equal(loaded[0], loaded[1]))
            finally:
                if saved is None:
                    os.environ.pop(CONTEXT_MASK_NPY_ENV, None)
                else:
                    os.environ[CONTEXT_MASK_NPY_ENV] = saved

    def test_env_loader_rejects_a_malformed_context_strength(self) -> None:
        """A bad strength fails at load time, not mid-sample."""
        alignment = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alignment.json"
            path.write_text(json.dumps(alignment), encoding="utf-8")
            saved = {
                key: os.environ.get(key)
                for key in (MASK_PATH_ENV, MODE_ENV, VOCAL_CNS_ENV, CONTEXT_STRENGTH_ENV)
            }
            try:
                os.environ[MASK_PATH_ENV] = str(path)
                os.environ[MODE_ENV] = "both"
                os.environ[VOCAL_CNS_ENV] = "0.6"
                os.environ[CONTEXT_STRENGTH_ENV] = "-0.5"
                with self.assertRaises(ValueError):
                    load_vocal_schedule_from_env(
                        torch.zeros(1, 3000, 32),
                        valid_timesteps=[1.0, 0.875, 0.75],
                        global_timestep=1.0,
                    )
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


class AnchorStrengthTest(unittest.TestCase):
    """Continuous vocal anchoring, instead of the hard clamp."""

    def test_strength_one_reproduces_the_hard_clamp(self) -> None:
        """The historical behaviour must stay bit-identical at strength 1.0."""
        xt = torch.randn(1, 4, 3)
        src = torch.randn(1, 4, 3)
        noise = torch.randn(1, 4, 3)
        mask = torch.tensor([[0.0], [1.0], [1.0], [0.0]]).view(1, 4, 1)
        blended = inject_source_path(xt, src, noise, mask, 0.5, strength=1.0)
        expected = torch.lerp(xt, source_path_at(src, noise, 0.5), mask)
        self.assertTrue(torch.equal(blended, expected))

    def test_zero_strength_is_an_exact_no_op(self) -> None:
        xt = torch.randn(1, 4, 3)
        out = inject_source_path(
            xt, torch.randn(1, 4, 3), torch.randn(1, 4, 3),
            torch.ones(1, 4, 1), 0.5, strength=0.0,
        )
        self.assertTrue(torch.equal(out, xt))

    def test_half_strength_keeps_half_of_the_generated_frame(self) -> None:
        """A vocal frame lands halfway between generated and source path."""
        xt = torch.zeros(1, 1, 1)
        src = torch.ones(1, 1, 1)
        noise = torch.zeros(1, 1, 1)
        mask = torch.ones(1, 1, 1)
        # source_path_at at t=0.5 is 0.5*0 + 0.5*1 = 0.5
        out = inject_source_path(xt, src, noise, mask, 0.5, strength=0.5)
        self.assertAlmostEqual(out.item(), 0.25, places=6)

    def test_accompaniment_frames_are_never_touched(self) -> None:
        xt = torch.randn(1, 3, 2)
        mask = torch.tensor([[0.0], [1.0], [0.0]]).view(1, 3, 1)
        out = inject_source_path(
            xt, torch.randn(1, 3, 2), torch.randn(1, 3, 2), mask, 0.5, strength=0.5
        )
        self.assertTrue(torch.equal(out[0, 0], xt[0, 0]))
        self.assertTrue(torch.equal(out[0, 2], xt[0, 2]))

    def test_env_defaults_to_the_hard_clamp_and_validates(self) -> None:
        saved = os.environ.pop(ANCHOR_STRENGTH_ENV, None)
        try:
            self.assertEqual(anchor_strength_from_env(), 1.0)
            os.environ[ANCHOR_STRENGTH_ENV] = "0.4"
            self.assertAlmostEqual(anchor_strength_from_env(), 0.4)
            os.environ[ANCHOR_STRENGTH_ENV] = "2.0"
            with self.assertRaises(ValueError):
                anchor_strength_from_env()
        finally:
            if saved is None:
                os.environ.pop(ANCHOR_STRENGTH_ENV, None)
            else:
                os.environ[ANCHOR_STRENGTH_ENV] = saved

    def test_spec_defaults_to_full_anchor(self) -> None:
        spec = _spec(torch.ones(1, 3, 1), 0.5, "both")
        self.assertEqual(spec.anchor_strength, 1.0)


if __name__ == "__main__":
    unittest.main()
