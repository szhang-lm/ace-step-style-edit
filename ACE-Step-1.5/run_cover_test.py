#!/usr/bin/env python3
"""Generate two short ACE-Step cover variants for a local source song."""

from pathlib import Path

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_AUDIO = PROJECT_ROOT / "test_audio" / "daisy_bell_30s.wav"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cover_test"

STYLES = {
    "jazz": "1920s jazz cover, warm upright piano, brushed drums, double bass, clear male vocal",
    "rock": "energetic rock cover, distorted electric guitar, punchy live drums, clear male vocal",
}


def main() -> None:
    """Load ACE-Step once and generate one deterministic result per target style."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    handler = AceStepHandler()
    status, success = handler.initialize_service(
        project_root=str(PROJECT_ROOT),
        config_path="acestep-v15-turbo",
        device="cuda:0",
        offload_to_cpu=False,
    )
    if not success:
        raise RuntimeError(status)
    logger.info(status)

    for index, (name, caption) in enumerate(STYLES.items()):
        params = GenerationParams(
            task_type="cover",
            src_audio=str(SOURCE_AUDIO),
            caption=caption,
            vocal_language="en",
            duration=30.0,
            audio_cover_strength=0.65,
            inference_steps=8,
            guidance_scale=1.0,
            seed=20261009 + index,
            thinking=False,
        )
        config = GenerationConfig(
            batch_size=1,
            audio_format="wav",
            use_random_seed=False,
            seeds=[20261009 + index],
        )
        result = generate_music(
            handler,
            None,
            params=params,
            config=config,
            save_dir=str(OUTPUT_ROOT / name),
        )
        if not result.success:
            raise RuntimeError(f"{name}: {result.status_message}")
        for audio in result.audios:
            logger.info("{} -> {}", name, audio.get("path", "in-memory"))


if __name__ == "__main__":
    main()
