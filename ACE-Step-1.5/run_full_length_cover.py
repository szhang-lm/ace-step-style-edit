#!/usr/bin/env python3
"""Generate full-length covers to test long-context consistency."""

import argparse
from pathlib import Path

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "test_audio" / "america_beautiful_male.opus"
OUT = ROOT / "outputs" / "multi_song_test" / "full_length"

LYRICS = """[Verse 1]
Oh, beautiful for spacious skies,
For amber waves of grain,
For purple mountain majesties
Above the fruited plain!
America! America!
God shed his grace on thee,
And crown thy good with brotherhood
From sea to shining sea.

[Instrumental Break]

[Verse 2]
Oh, beautiful for patriot dream
That sees beyond the years
Thine alabaster cities gleam,
Undimmed by human tears!
America! America!
God shed his grace on thee,
And crown thy good with brotherhood
From sea to shining sea!
"""

STYLES = {
    "jazz": "smooth jazz, upright piano, brushed drums, double bass, warm saxophone",
    "electronic": "modern electronic pop, synthesizers, electronic drums, deep bass, polished production",
}


def main() -> None:
    """Generate one selected full-length style using official lyrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=STYLES, required=True)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lyrics.txt").write_text(LYRICS, encoding="utf-8")

    handler = AceStepHandler()
    status, success = handler.initialize_service(
        project_root=str(ROOT),
        config_path="acestep-v15-turbo",
        device="cuda:0",
        offload_to_cpu=False,
    )
    if not success:
        raise RuntimeError(status)

    seed = 20263001 if args.style == "jazz" else 20263002
    result = generate_music(
        handler,
        None,
        params=GenerationParams(
            task_type="cover",
            src_audio=str(SOURCE),
            reference_audio=str(SOURCE),
            caption=(
                f"{STYLES[args.style]}, preserve the original male singer, clear prominent "
                "male lead vocal with the same vocal identity, preserve the original melody, "
                "lyrics, timing, and instrumental break"
            ),
            lyrics=LYRICS,
            vocal_language="en",
            duration=225.5,
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
        save_dir=str(OUT / f"raw_{args.style}"),
    )
    if not result.success:
        raise RuntimeError(result.status_message)
    generated = Path(result.audios[0]["path"])
    output_path = OUT / f"america_beautiful_full_{args.style}.wav"
    output_path.write_bytes(generated.read_bytes())
    logger.info("Saved {}", output_path)


if __name__ == "__main__":
    main()
