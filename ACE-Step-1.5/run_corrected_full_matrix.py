#!/usr/bin/env python3
"""Regenerate full-song covers with lyrics calibrated to each source audio."""

import argparse
import json
from pathlib import Path

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "multi_song_test"
SOURCE_DIR = BASE / "sources"
LYRIC_DIR = BASE / "03_corrected_lyrics"
OUTPUT_DIR = BASE / "04_corrected_covers"

LYRICS = {
    "chinese_female": """[Verse]
让这歌陪我走过山海，每一句话都不会被掩埋。
无论时间把故事怎样更改，我还记得你眼里的期待。
夜晚的灯照亮旧的站台，离别的音写进温柔节拍。
等下一场花在春天盛开。
远方的云慢慢向前走，像我们曾经许下的以后。
让这首歌陪我走过更改，我还记得你眼里的期待。
无论时间把故事怎样更改，我还记得你眼里的期待。
让这旋律穿过人潮大海，告诉未来我们依然存在。
我们依然在，依然存在，我们依然存在。
""",
    "english_female": """[Verse]
O say can you see, by the dawn's early light,
What so proudly we hailed at the twilight's last gleaming?
Whose broad stripes and bright stars through the perilous fight,
O'er the ramparts we watched, were so gallantly streaming?
And the rocket's red glare, the bombs bursting in air,
Gave proof through the night that our flag was still there;
O say does that star-spangled banner yet wave
O'er the land of the free and the home of the brave?
""",
    "english_male": """[Instrumental Intro]

[Verse]
O beautiful for spacious skies,
For amber waves of grain,
For purple mountain majesties
Above the fruited plain!
America! America!
God shed His grace on thee,
And crown thy good with brotherhood
From sea to shining sea.

[Instrumental Break]
""",
}

SONGS = {
    "chinese_female": {
        "path": SOURCE_DIR / "chinese_female_120s.wav",
        "language": "zh",
        "duration": 120.0,
        "voice": "female",
    },
    "english_female": {
        "path": SOURCE_DIR / "english_female_80s.wav",
        "language": "en",
        "duration": 80.0,
        "voice": "female",
    },
    "english_male": {
        "path": SOURCE_DIR / "english_male_120s.wav",
        "language": "en",
        "duration": 120.0,
        "voice": "male",
    },
}

STYLES = {
    "jazz": "smooth jazz, upright piano, brushed drums, double bass, warm saxophone",
    "rock": "modern rock, distorted electric guitar, powerful bass, energetic live drums",
    "electronic": "modern electronic pop, synthesizers, electronic drums, deep bass, polished production",
    "folk": "intimate acoustic folk, acoustic guitar, light percussion, natural warm arrangement",
}


def parse_args() -> argparse.Namespace:
    """Parse matrix shard options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Generate one shard of corrected full-song covers."""
    args = parse_args()
    LYRIC_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for song_name, lyrics in LYRICS.items():
        (LYRIC_DIR / f"{song_name}.txt").write_text(lyrics, encoding="utf-8")

    cases = [(song, style) for song in SONGS for style in STYLES]
    cases = cases[args.shard_index :: args.shard_count]

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
    for case_index, (song_name, style_name) in enumerate(cases):
        song = SONGS[song_name]
        voice = song["voice"]
        seed = 20264000 + args.shard_index * 100 + case_index
        result = generate_music(
            handler,
            None,
            params=GenerationParams(
                task_type="cover",
                src_audio=str(song["path"]),
                reference_audio=str(song["path"]),
                caption=(
                    f"{STYLES[style_name]}, preserve the original {voice} singer and vocal "
                    "identity, preserve the original melody, lyric order, pauses, and structure"
                ),
                lyrics=LYRICS[song_name],
                vocal_language=song["language"],
                duration=song["duration"],
                audio_cover_strength=0.8,
                cover_noise_strength=0.6,
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
            save_dir=str(OUTPUT_DIR / "raw" / f"{song_name}_{style_name}"),
        )
        if not result.success:
            raise RuntimeError(f"{song_name}/{style_name}: {result.status_message}")

        generated = Path(result.audios[0]["path"])
        output_path = OUTPUT_DIR / f"{song_name}_{style_name}.wav"
        output_path.write_bytes(generated.read_bytes())
        records.append(
            {
                "song": song_name,
                "style": style_name,
                "source": str(song["path"]),
                "lyrics": str(LYRIC_DIR / f"{song_name}.txt"),
                "output": str(output_path),
                "duration": song["duration"],
                "seed": seed,
            }
        )
        logger.info("Saved {}", output_path)

    (OUTPUT_DIR / f"metadata_shard{args.shard_index}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
