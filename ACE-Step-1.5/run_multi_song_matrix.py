#!/usr/bin/env python3
"""Generate a multi-song, multi-style ACE-Step Cover evaluation matrix."""

import argparse
import json
from pathlib import Path

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "outputs" / "multi_song_test" / "sources"
OUT = ROOT / "outputs" / "multi_song_test" / "covers"

STAR_LYRICS = """[Verse]
O say can you see, by the dawn's early light,
What so proudly we hailed at the twilight's last gleaming,
Whose broad stripes and bright stars through the perilous fight,
O'er the ramparts we watched, were so gallantly streaming?
And the rocket's red glare, the bombs bursting in air,
Gave proof through the night that our flag was still there;
O say does that star-spangled banner yet wave
O'er the land of the free and the home of the brave?
"""

AMERICA_LYRICS = """[Verse 1]
O beautiful for spacious skies,
For amber waves of grain,
For purple mountain majesties
Above the fruited plain!
America! America!
God shed His grace on thee,
And crown thy good with brotherhood
From sea to shining sea!

[Verse 2]
O beautiful for pilgrim feet,
Whose stern, impassioned stress
A thoroughfare for freedom beat
Across the wilderness!
America! America!
God mend thine every flaw,
Confirm thy soul in self-control,
Thy liberty in law!
"""

CHINESE_LYRICS = (SOURCE_DIR / "chinese_female_lyrics.txt").read_text(encoding="utf-8")

SONGS = {
    "english_female": {
        "path": SOURCE_DIR / "english_female_80s.wav",
        "lyrics": STAR_LYRICS,
        "language": "en",
        "duration": 80.0,
        "voice": "female",
    },
    "english_male": {
        "path": SOURCE_DIR / "english_male_120s.wav",
        "lyrics": AMERICA_LYRICS,
        "language": "en",
        "duration": 120.0,
        "voice": "male",
    },
    "chinese_female": {
        "path": SOURCE_DIR / "chinese_female_120s.wav",
        "lyrics": CHINESE_LYRICS,
        "language": "zh",
        "duration": 120.0,
        "voice": "female",
    },
}

STYLES = {
    "jazz": "smooth jazz, upright piano, brushed drums, double bass, warm saxophone",
    "rock": "modern rock, distorted electric guitar, powerful bass, energetic live drums",
    "electronic": "modern electronic pop, synthesizers, electronic drums, deep bass, polished production",
    "folk": "intimate acoustic folk, acoustic guitar, light percussion, natural warm arrangement",
}


def parse_args() -> argparse.Namespace:
    """Parse GPU-sharding arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Generate one assigned quarter of the evaluation matrix."""
    args = parse_args()
    cases = [(song, style) for song in SONGS for style in STYLES]
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
    for case_number, (song_name, style_name) in enumerate(cases):
        song = SONGS[song_name]
        voice = song["voice"]
        caption = (
            f"{STYLES[style_name]}, preserve the original {voice} singer, "
            f"clear prominent {voice} lead vocal, preserve the original melody and lyrics"
        )
        seed = 20262000 + args.shard_index * 100 + case_number
        result = generate_music(
            handler,
            None,
            params=GenerationParams(
                task_type="cover",
                src_audio=str(song["path"]),
                reference_audio=str(song["path"]),
                caption=caption,
                lyrics=song["lyrics"],
                vocal_language=song["language"],
                duration=song["duration"],
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
            save_dir=str(OUT / "raw" / f"{song_name}_{style_name}"),
        )
        if not result.success:
            raise RuntimeError(f"{song_name}/{style_name}: {result.status_message}")
        generated = Path(result.audios[0]["path"])
        output_path = OUT / f"{song_name}_{style_name}.wav"
        output_path.write_bytes(generated.read_bytes())
        records.append(
            {
                "song": song_name,
                "style": style_name,
                "path": str(output_path),
                "duration": song["duration"],
                "cover_strength": 0.8,
                "reference_audio": True,
                "seed": seed,
            }
        )
        logger.info("Saved {}", output_path)

    metadata_path = OUT / f"metadata_shard{args.shard_index}.json"
    metadata_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
