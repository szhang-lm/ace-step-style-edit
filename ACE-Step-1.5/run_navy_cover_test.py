#!/usr/bin/env python3
"""Create cover variants from a clear U.S. Navy Band vocal recording."""

from pathlib import Path

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "test_audio" / "star_spangled_banner_30s.wav"
OUT = ROOT / "outputs" / "navy_cover_test"
LYRICS = """[Verse]
O say can you see, by the dawn's early light,
What so proudly we hailed at the twilight's last gleaming,
Whose broad stripes and bright stars through the perilous fight,
O'er the ramparts we watched, were so gallantly streaming.
"""

STYLES = {
    "jazz": "smooth jazz cover, prominent clear solo lead vocal, upright piano, brushed drums, double bass",
    "rock": "powerful modern rock cover, prominent clear solo lead vocal, electric guitar, bass, live drums",
}


def main() -> None:
    """Load ACE-Step and generate two styles from the same clean source."""
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

    for index, (name, caption) in enumerate(STYLES.items()):
        seed = 20261020 + index
        result = generate_music(
            handler,
            None,
            params=GenerationParams(
                task_type="cover",
                src_audio=str(SOURCE),
                caption=caption,
                lyrics=LYRICS,
                vocal_language="en",
                duration=30.0,
                audio_cover_strength=0.8,
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
        final_path = OUT / f"star_spangled_{name}.wav"
        final_path.write_bytes(generated.read_bytes())
        logger.info("{} -> {}", name, final_path)


if __name__ == "__main__":
    main()
