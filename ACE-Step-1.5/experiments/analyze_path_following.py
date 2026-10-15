"""Decompose generated style paths into aligned strength and orthogonal drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


KINDS = ("velocity", "predicted_clean", "state")


def parse_args() -> argparse.Namespace:
    """Parse trajectory inputs and output path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--real-cover", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    """Load a saved trajectory on CPU."""
    return torch.load(path, map_location="cpu", weights_only=True)


def decompose(generated: torch.Tensor, real: torch.Tensor) -> dict[str, float]:
    """Project generated offset onto the real-cover offset."""
    generated = generated.reshape(-1).float()
    real = real.reshape(-1).float()
    real_energy = torch.dot(real, real).clamp_min(1e-12)
    generated_energy = torch.dot(generated, generated).clamp_min(1e-12)
    coefficient = torch.dot(generated, real) / real_energy
    projection = coefficient * real
    orthogonal = generated - projection
    cosine = torch.dot(generated, real) / torch.sqrt(
        generated_energy * real_energy
    )
    return {
        "cosine": float(cosine),
        # 1.0 means the component along the true path has the same magnitude.
        "parallel_strength": float(coefficient),
        "generated_to_real_norm": float(
            torch.sqrt(generated_energy / real_energy)
        ),
        # Relative to the real path: 0 means no off-path motion.
        "orthogonal_to_real_norm": float(
            torch.linalg.vector_norm(orthogonal) / torch.sqrt(real_energy)
        ),
        # Fraction of generated energy explained by the real path direction.
        "direction_explained_fraction": float(cosine.square()),
    }


def main() -> None:
    """Compute and save path-following decomposition metrics."""
    args = parse_args()
    trajectories = {
        "original": load(args.original),
        "real_cover": load(args.real_cover),
        "generated": load(args.generated),
    }
    count = min(len(item["trace"]) for item in trajectories.values())
    rows = []
    for step in range(count):
        original = trajectories["original"]["trace"][step]
        real = trajectories["real_cover"]["trace"][step]
        generated = trajectories["generated"]["trace"][step]
        timesteps = [original["timestep"], real["timestep"], generated["timestep"]]
        if max(timesteps) - min(timesteps) > 1e-6:
            raise ValueError(f"Timestep mismatch at step {step}: {timesteps}")
        row: dict[str, object] = {"step": step, "timestep": timesteps[0]}
        for kind in KINDS:
            real_offset = real[kind] - original[kind]
            generated_offset = generated[kind] - original[kind]
            row[kind] = decompose(generated_offset, real_offset)
        rows.append(row)

    mean = {
        kind: {
            metric: sum(row[kind][metric] for row in rows) / len(rows)
            for metric in rows[0][kind]
        }
        for kind in KINDS
    }
    result = {
        "inputs": {
            "original": str(args.original),
            "real_cover": str(args.real_cover),
            "generated": str(args.generated),
        },
        "steps": rows,
        "mean": mean,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(mean, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
