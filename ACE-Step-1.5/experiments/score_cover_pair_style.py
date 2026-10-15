"""Score a source, real cover, and generated edit against style prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_nonuniform_noise import ClapEmbedder, load_audio, to_mono_48k


PROMPTS = {
    "pop_funk": "upbeat pop funk with acoustic guitar and clean male vocals",
    "rock_metal": (
        "aggressive alternative metal and rap rock with distorted electric "
        "guitars, heavy drums and forceful male vocals"
    ),
    "jazz": "smooth vocal jazz with piano, upright bass and brushed drums",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cover", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--generated-alt", type=Path)
    parser.add_argument("--source-prompt")
    parser.add_argument("--target-prompt")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clap = ClapEmbedder()
    prompts = PROMPTS
    if args.source_prompt and args.target_prompt:
        prompts = {
            "source_style": args.source_prompt,
            "target_style": args.target_prompt,
        }
    text_features = {}
    for name, prompt in prompts.items():
        inputs = clap.processor(text=[prompt], return_tensors="pt")
        inputs = {key: value.to(clap.device) for key, value in inputs.items()}
        feature = clap.model.get_text_features(**inputs).squeeze(0).float().detach().cpu().numpy()
        text_features[name] = feature / (np.linalg.norm(feature) + 1e-12)

    audio_features = {}
    rows = []
    entries = [
        ("source", args.source),
        ("real_cover", args.cover),
        ("generated", args.generated),
    ]
    if args.generated_alt is not None:
        entries.append(("generated_alt", args.generated_alt))
    for name, path in entries:
        audio, rate = load_audio(path)
        np.random.seed(0)
        torch.manual_seed(0)
        feature = clap.embed(to_mono_48k(audio, rate))
        feature = feature / (np.linalg.norm(feature) + 1e-12)
        audio_features[name] = feature
        rows.append(
            {
                "name": name,
                "path": str(path),
                "style_scores": {
                    style: float(np.dot(feature, text_feature))
                    for style, text_feature in text_features.items()
                },
            }
        )
    similarities = {
        "source_to_real_cover": float(
            np.dot(audio_features["source"], audio_features["real_cover"])
        ),
        "source_to_generated": float(
            np.dot(audio_features["source"], audio_features["generated"])
        ),
        "generated_to_real_cover": float(
            np.dot(audio_features["generated"], audio_features["real_cover"])
        ),
    }
    if "generated_alt" in audio_features:
        similarities.update(
            {
                "source_to_generated_alt": float(
                    np.dot(audio_features["source"], audio_features["generated_alt"])
                ),
                "generated_alt_to_real_cover": float(
                    np.dot(audio_features["generated_alt"], audio_features["real_cover"])
                ),
                "generated_to_generated_alt": float(
                    np.dot(audio_features["generated"], audio_features["generated_alt"])
                ),
            }
        )
    result = {"prompts": prompts, "rows": rows, "audio_similarities": similarities}
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
