"""Diagnostic upper bound: nudge cover generation along a real-cover path."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Parse generation and oracle-path parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--original-path", type=Path, required=True)
    parser.add_argument("--real-cover-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cover-noise-strength", type=float, default=0.2)
    return parser.parse_args()


def load_trace(path: Path) -> list[dict]:
    """Load a saved denoising trace on CPU."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["trace"]


def main() -> None:
    """Generate one cover while adding a scaled real-cover velocity offset."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = load_trace(args.original_path)
    real_cover = load_trace(args.real_cover_path)
    count = min(len(original), len(real_cover))
    reference_deltas = []
    for step in range(count):
        if abs(original[step]["timestep"] - real_cover[step]["timestep"]) > 1e-6:
            raise ValueError(f"Reference timestep mismatch at step {step}")
        reference_deltas.append(
            real_cover[step]["velocity"].float()
            - original[step]["velocity"].float()
        )
    handler = AceStepHandler()
    status, initialized = handler.initialize_service(
        project_root=str(REPO_ROOT),
        config_path="acestep-v15-turbo",
        device="mps",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
        use_mlx_dit=False,
    )
    if not initialized:
        raise RuntimeError(status)
    assist_log: list[dict] = []
    corrected_trace: list[dict] = []
    assist_step = 0
    capture_step = 0
    def apply_assist(_module, _inputs, kwargs, output):
        nonlocal assist_step
        if assist_step >= len(reference_deltas):
            raise RuntimeError("Model used more denoising steps than the reference path")
        timestep = float(kwargs["timestep"].detach().float().mean().cpu())
        reference_timestep = float(original[assist_step]["timestep"])
        if abs(timestep - reference_timestep) > 1e-5:
            raise RuntimeError(
                f"Timestep mismatch at {assist_step}: {timestep} vs {reference_timestep}"
            )
        velocity = output[0]
        delta = reference_deltas[assist_step].to(
            device=velocity.device, dtype=velocity.dtype
        )
        if tuple(delta.shape) != tuple(velocity.shape[1:]):
            raise RuntimeError(
                f"Reference shape {tuple(delta.shape)} != model {tuple(velocity.shape[1:])}"
            )
        corrected = velocity + args.alpha * delta.unsqueeze(0)
        assist_log.append(
            {
                "step": assist_step,
                "timestep": timestep,
                "raw_velocity_norm": float(velocity.float().norm().cpu()),
                "reference_delta_norm": float(delta.float().norm().cpu()),
                "applied_delta_norm": float(
                    (args.alpha * delta.float()).norm().cpu()
                ),
            }
        )
        assist_step += 1
        return (corrected, *output[1:])
    def capture(_module, _inputs, kwargs, output):
        nonlocal capture_step
        state = kwargs["hidden_states"].detach().cpu()[0].float()
        timestep = float(kwargs["timestep"].detach().float().mean().cpu())
        velocity = output[0].detach().cpu()[0].float()
        corrected_trace.append(
            {
                "step": capture_step,
                "timestep": timestep,
                "state": state,
                "velocity": velocity,
                "predicted_clean": state - timestep * velocity,
            }
        )
        capture_step += 1
    assist_handle = handler.model.decoder.register_forward_hook(
        apply_assist, with_kwargs=True
    )
    capture_handle = handler.model.decoder.register_forward_hook(
        capture, with_kwargs=True
    )
    params = GenerationParams(
        task_type="cover",
        src_audio=str(args.source),
        caption=args.caption,
        lyrics="",
        instrumental=False,
        vocal_language="en",
        audio_cover_strength=1.0,
        cover_noise_strength=args.cover_noise_strength,
        inference_steps=args.steps,
        guidance_scale=1.0,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        seed=args.seed,
    )
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=False,
        seeds=[args.seed],
        audio_format="wav",
    )
    started = time.time()
    try:
        result = generate_music(
            dit_handler=handler,
            llm_handler=LLMHandler(),
            params=params,
            config=config,
            save_dir=str(args.output_dir),
        )
    finally:
        capture_handle.remove()
        assist_handle.remove()
    if not result.success:
        raise RuntimeError(result.error or result.status_message)
    if assist_step != count:
        raise RuntimeError(f"Used {assist_step} reference steps, expected {count}")
    summary = {
        "source": str(args.source),
        "caption": args.caption,
        "seed": args.seed,
        "steps_requested": args.steps,
        "steps_executed": assist_step,
        "cover_noise_strength": args.cover_noise_strength,
        "alpha": args.alpha,
        "reference_original": str(args.original_path),
        "reference_real_cover": str(args.real_cover_path),
        "generated_audio": result.audios,
        "elapsed_seconds": time.time() - started,
        "assist": assist_log,
    }
    (args.output_dir / "oracle_assist.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    torch.save(
        {
            "source": str(args.source),
            "caption": args.caption,
            "seed": args.seed,
            "alpha": args.alpha,
            "trace": corrected_trace,
        },
        args.output_dir / "trajectory_tensors.pt",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__":
    main()
