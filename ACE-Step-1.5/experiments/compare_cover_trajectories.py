"""Compare self-denoising trajectories for original, real cover, and edit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--real-cover", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.reshape(-1).float()
    second = second.reshape(-1).float()
    return float(torch.dot(first, second) / (first.norm() * second.norm() + 1e-8))


def nmse(first: torch.Tensor, second: torch.Tensor) -> float:
    error = (first.float() - second.float()).square().mean()
    scale = second.float().square().mean() + 1e-8
    return float(error / scale)


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    args = parse_args()
    paths = {
        "original": load(args.original),
        "real_cover": load(args.real_cover),
        "generated": load(args.generated),
    }
    step_count = min(len(item["trace"]) for item in paths.values())
    rows = []
    for step in range(step_count):
        original = paths["original"]["trace"][step]
        real = paths["real_cover"]["trace"][step]
        generated = paths["generated"]["trace"][step]
        timesteps = [original["timestep"], real["timestep"], generated["timestep"]]
        if max(timesteps) - min(timesteps) > 1e-6:
            raise ValueError(f"Timestep mismatch at step {step}: {timesteps}")
        rows.append(
            {
                "step": step,
                "timestep": timesteps[0],
                "velocity_cosine_real_original": cosine(
                    real["velocity"], original["velocity"]
                ),
                "velocity_cosine_real_generated": cosine(
                    real["velocity"], generated["velocity"]
                ),
                "clean_cosine_real_original": cosine(
                    real["predicted_clean"], original["predicted_clean"]
                ),
                "clean_cosine_real_generated": cosine(
                    real["predicted_clean"], generated["predicted_clean"]
                ),
                "state_nmse_real_original": nmse(real["state"], original["state"]),
                "state_nmse_real_generated": nmse(real["state"], generated["state"]),
                "style_velocity_cosine": cosine(
                    real["velocity"] - original["velocity"],
                    generated["velocity"] - original["velocity"],
                ),
                "style_clean_cosine": cosine(
                    real["predicted_clean"] - original["predicted_clean"],
                    generated["predicted_clean"] - original["predicted_clean"],
                ),
                "style_state_cosine": cosine(
                    real["state"] - original["state"],
                    generated["state"] - original["state"],
                ),
            }
        )
    summary = {
        "inputs": {
            name: {
                "trajectory": str(path),
                "audio": data["source"],
                "caption": data["caption"],
                "seed": data["seed"],
            }
            for name, (path, data) in zip(
                paths,
                [
                    (args.original, paths["original"]),
                    (args.real_cover, paths["real_cover"]),
                    (args.generated, paths["generated"]),
                ],
            )
        },
        "steps": rows,
        "mean": {
            key: sum(row[key] for row in rows) / len(rows)
            for key in rows[0]
            if key not in {"step", "timestep"}
        },
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
