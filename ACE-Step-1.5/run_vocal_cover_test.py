#!/usr/bin/env python3
"""Test whether ACE-Step cover preserves vocals with explicit lyrics."""

from pathlib import Path

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "vocal_cover_test"
LYRICS = """[Verse]
Morning light is shining through the window
I can hear the city waking slow
Every road is calling me to follow
Today I know exactly where to go

[Chorus]
Sing with me under the open sky
Hold this moment as the world goes by
Sing with me, let every heartbeat fly
We will keep this little song alive
"""


def generate(handler: AceStepHandler, params: GenerationParams, folder: str) -> Path:
    """Generate one WAV and return its saved path."""
    result = generate_music(
        handler,
        None,
        params=params,
        config=GenerationConfig(
            batch_size=1,
            audio_format="wav",
            use_random_seed=False,
            seeds=[params.seed],
        ),
        save_dir=str(OUT / folder),
    )
    if not result.success:
        raise RuntimeError(result.status_message)
    path = Path(result.audios[0]["path"])
    logger.info("{} -> {}", folder, path)
    return path


def main() -> None:
    """Generate a clean vocal source, then cover it with explicit lyrics."""
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

    source = generate(
        handler,
        GenerationParams(
            task_type="text2music",
            caption="modern acoustic pop, clean prominent female lead vocal, gentle piano and guitar",
            lyrics=LYRICS,
            vocal_language="en",
            duration=30.0,
            bpm=92,
            keyscale="C major",
            timesignature="4",
            inference_steps=8,
            guidance_scale=1.0,
            seed=20261011,
            thinking=False,
        ),
        "source_pop",
    )

    cover = generate(
        handler,
        GenerationParams(
            task_type="cover",
            src_audio=str(source),
            caption="smooth jazz cover, prominent clear female lead vocal, upright piano, brushed drums, double bass",
            lyrics=LYRICS,
            vocal_language="en",
            duration=30.0,
            audio_cover_strength=0.8,
            inference_steps=8,
            guidance_scale=1.0,
            seed=20261012,
            thinking=False,
        ),
        "cover_jazz_with_lyrics",
    )

    (OUT / "source_pop.wav").write_bytes(source.read_bytes())
    (OUT / "cover_jazz_with_lyrics.wav").write_bytes(cover.read_bytes())


if __name__ == "__main__":
    main()
