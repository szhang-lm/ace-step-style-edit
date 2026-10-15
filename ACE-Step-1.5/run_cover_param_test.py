#!/usr/bin/env python3
"""Cover 参数对照：验证 cover_noise_strength / audio_cover_strength 对歌词保留的影响。

固定用 english_male + rock（原跑法覆盖率仅 18.9%），跑几个参数组合。
"""
from pathlib import Path

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "multi_song_test"
SOURCE = BASE / "sources" / "english_male_120s.wav"
LYRICS = (BASE / "03_corrected_lyrics" / "english_male.txt").read_text()
OUT = BASE / "param_test"
STYLE = "modern rock, distorted electric guitar, powerful bass, energetic live drums"

# (name, audio_cover_strength, cover_noise_strength)
CASES = [
    ("acs0.8_cns0.0", 0.8, 0.0),   # 原跑法（基准）
    ("acs0.8_cns0.5", 0.8, 0.5),
    ("acs0.8_cns0.8", 0.8, 0.8),
    ("acs1.0_cns0.8", 1.0, 0.8),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    handler = AceStepHandler()
    status, success = handler.initialize_service(
        project_root=str(ROOT), config_path="acestep-v15-turbo",
        device="cuda:0", offload_to_cpu=False,
    )
    if not success:
        raise RuntimeError(status)

    for name, acs, cns in CASES:
        seed = 20265000
        result = generate_music(
            handler, None,
            params=GenerationParams(
                task_type="cover",
                src_audio=str(SOURCE),
                reference_audio=str(SOURCE),
                caption=(f"{STYLE}, preserve the original male singer and vocal "
                         "identity, preserve the original melody, lyric order, pauses, and structure"),
                lyrics=LYRICS,
                vocal_language="en",
                duration=120.0,
                audio_cover_strength=acs,
                cover_noise_strength=cns,
                inference_steps=8,
                guidance_scale=1.0,
                seed=seed,
                thinking=False,
            ),
            config=GenerationConfig(batch_size=1, audio_format="wav",
                                    use_random_seed=False, seeds=[seed]),
            save_dir=str(OUT / name),
        )
        if not result.success:
            logger.error("{} failed: {}", name, result.status_message)
            continue
        generated = Path(result.audios[0]["path"])
        (OUT / f"{name}.wav").write_bytes(generated.read_bytes())
        logger.info("Saved {} -> {}", name, OUT / f"{name}.wav")


if __name__ == "__main__":
    main()
