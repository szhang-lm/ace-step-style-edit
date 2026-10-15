#!/usr/bin/env python3
"""Run a sharded ACE-Step Cover parameter sweep for thesis problem discovery."""

import argparse
import json
from pathlib import Path

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "test_audio" / "star_spangled_complete_phrase.wav"
OUT = ROOT / "outputs" / "cover_sweep"
LYRICS = """[Verse]
O say can you see, by the dawn's early light,
What so proudly we hailed at the twilight's last gleaming,
Whose broad stripes and bright stars through the perilous fight,
O'er the ramparts we watched, were so gallantly streaming.
"""

STYLES = {
    "jazz": (
        "smooth jazz arrangement, preserve the original female solo singer, clearly female "
        "lead vocal, upright piano, brushed drums, double bass"
    ),
    "rock": (
        "powerful modern rock arrangement, preserve the original female solo singer, clearly "
        "female lead vocal, distorted electric guitar, bass, live drums"
    ),
}
STRENGTHS = (0.6, 0.8, 0.95, 1.0)
REFERENCE_MODES = ("none", "full_mix")
SEEDS = (20261101, 20261102)


def parse_args() -> argparse.Namespace:
    """Parse sharding options for multi-GPU execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate the assigned subset and save deterministic files plus metadata."""
    args = parse_args()
    cases = [
        (style, strength, reference_mode)
        for style in STYLES
        for strength in STRENGTHS
        for reference_mode in REFERENCE_MODES
    ]
    cases = cases[args.shard_index :: args.shard_count]
    OUT.mkdir(parents=True, exist_ok=True)

    handler = AceStepHandler()
    status, success = handler.initialize_service(
        project_root=str(ROOT),
        config_path="acestep-v15-turbo",
        device="cuda:0",
        offload_to_cpu=False,
    )
    if not success:
        raise RuntimeError(status)

    records = []
    for style, strength, reference_mode in cases:
        case_name = f"{style}_s{strength:.2f}_{reference_mode}"
        case_dir = OUT / case_name
        result = generate_music(
            handler,
            None,
            params=GenerationParams(
                task_type="cover",
                src_audio=str(SOURCE),
                reference_audio=str(SOURCE) if reference_mode == "full_mix" else None,
                caption=STYLES[style],
                lyrics=LYRICS,
                vocal_language="en",
                duration=39.5,
                audio_cover_strength=strength,
                inference_steps=8,
                guidance_scale=1.0,
                seed=SEEDS[0],
                thinking=False,
            ),
            config=GenerationConfig(
                batch_size=len(SEEDS),
                audio_format="wav",
                use_random_seed=False,
                seeds=list(SEEDS),
            ),
            save_dir=str(case_dir),
        )
        if not result.success:
            raise RuntimeError(f"{case_name}: {result.status_message}")

        for seed, audio in zip(SEEDS, result.audios):
            generated = Path(audio["path"])
            final_path = OUT / f"{case_name}_seed{seed}.wav"
            final_path.write_bytes(generated.read_bytes())
            records.append(
                {
                    "path": str(final_path),
                    "style": style,
                    "cover_strength": strength,
                    "reference_mode": reference_mode,
                    "seed": seed,
                }
            )
            logger.info("Saved {}", final_path)

    metadata_path = OUT / f"metadata_shard{args.shard_index}.json"
    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
