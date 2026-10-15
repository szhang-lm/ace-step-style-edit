#!/usr/bin/env python3
"""Test strict lyric alignment and female-vocal preservation in Cover mode."""

from pathlib import Path

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "test_audio" / "star_spangled_complete_phrase.wav"
OUT = ROOT / "outputs" / "strict_cover_test"
LYRICS = """[Verse]
O say can you see, by the dawn's early light,
What so proudly we hailed at the twilight's last gleaming,
Whose broad stripes and bright stars through the perilous fight,
O'er the ramparts we watched, were so gallantly streaming.
"""


def main() -> None:
    """Generate strict covers with and without an explicit timbre reference."""
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

    cases = {
        "female_prompt": None,
        "female_timbre_reference": str(SOURCE),
    }
    for index, (name, reference_audio) in enumerate(cases.items()):
        seed = 20261030 + index
        result = generate_music(
            handler,
            None,
            params=GenerationParams(
                task_type="cover",
                src_audio=str(SOURCE),
                reference_audio=reference_audio,
                caption=(
                    "smooth jazz arrangement, preserve the original female solo singer, "
                    "clearly female lead vocal with the same vocal timbre, upright piano, "
                    "brushed drums, double bass"
                ),
                lyrics=LYRICS,
                vocal_language="en",
                duration=39.5,
                audio_cover_strength=0.95,
                inference_steps=8,
                guidance_scale=1.0,
                seed=seed,
                thinking=False,
            ),
            config=GenerationConfig(
                batch_size=1,
                audio_format="wav",
                use_random_seed=False,
                seeds=[seed],
            ),
            save_dir=str(OUT / name),
        )
        if not result.success:
            raise RuntimeError(f"{name}: {result.status_message}")
        generated = Path(result.audios[0]["path"])
        final_path = OUT / f"{name}.wav"
        final_path.write_bytes(generated.read_bytes())
        logger.info("{} -> {}", name, final_path)


if __name__ == "__main__":
    main()
