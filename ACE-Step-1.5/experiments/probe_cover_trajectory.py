"""Probe whether an ACE-Step edit trajectory approaches a real cover latent."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import soundfile as sf
import torch

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cover-noise-strength", type=float, default=0.2)
    return parser.parse_args()


def load_audio(path: Path) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 48000:
        raise ValueError(f"Expected 48 kHz audio, got {sample_rate}: {path}")
    return torch.from_numpy(audio.T.copy())


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float().reshape(-1)
    second = second.float().reshape(-1)
    return float(torch.dot(first, second) / (first.norm() * second.norm() + 1e-8))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    source_audio = load_audio(args.source)
    target_audio = load_audio(args.target)
    with torch.inference_mode(), handler._load_model_context("vae"):
        torch.manual_seed(args.seed)
        source_latent = handler._encode_audio_to_latents(source_audio).detach().cpu()
        torch.manual_seed(args.seed + 1)
        target_latent = handler._encode_audio_to_latents(target_audio).detach().cpu()
    length = min(source_latent.shape[0], target_latent.shape[0])
    source_latent = source_latent[:length]
    target_latent = target_latent[:length]
    oracle = target_latent - source_latent

    trace = []
    tensor_trace = []

    def capture(_module, _args, kwargs, output):
        xt = kwargs["hidden_states"].detach().cpu()[0, :length]
        timestep = float(kwargs["timestep"].detach().float().mean().cpu())
        velocity = output[0].detach().cpu()[0, :length]
        predicted_clean = xt - timestep * velocity
        predicted_edit = predicted_clean - source_latent
        tensor_trace.append(
            {
                "timestep": timestep,
                "state": xt.float(),
                "velocity": velocity.float(),
                "predicted_clean": predicted_clean.float(),
            }
        )
        oracle_energy = float(oracle.float().square().mean()) + 1e-8
        trace.append(
            {
                "step": len(trace),
                "timestep": timestep,
                "predicted_to_source_mse": float(
                    (predicted_clean.float() - source_latent.float()).square().mean()
                ),
                "predicted_to_cover_mse": float(
                    (predicted_clean.float() - target_latent.float()).square().mean()
                ),
                "edit_oracle_cosine": cosine(predicted_edit, oracle),
                "edit_oracle_projection": float(
                    (predicted_edit.float() * oracle.float()).mean() / oracle_energy
                ),
            }
        )

    hook = handler.model.decoder.register_forward_hook(capture, with_kwargs=True)
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
        hook.remove()
    if not result.success:
        raise RuntimeError(result.error or result.status_message)

    summary = {
        "source": str(args.source),
        "real_cover": str(args.target),
        "caption": args.caption,
        "seed": args.seed,
        "steps_requested": args.steps,
        "cover_noise_strength": args.cover_noise_strength,
        "latent_frames": length,
        "source_cover_latent_cosine": cosine(source_latent, target_latent),
        "source_cover_latent_mse": float(
            (source_latent.float() - target_latent.float()).square().mean()
        ),
        "trace": trace,
        "generated_audio": result.audios,
        "elapsed_seconds": time.time() - started,
    }
    output_path = args.output_dir / "trajectory_probe.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    torch.save(
        {
            "source": str(args.source),
            "caption": args.caption,
            "seed": args.seed,
            "cover_noise_strength": args.cover_noise_strength,
            "source_latent": source_latent.float(),
            "trace": tensor_trace,
        },
        args.output_dir / "trajectory_tensors.pt",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
