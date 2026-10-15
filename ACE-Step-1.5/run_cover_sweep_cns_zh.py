#!/usr/bin/env python3
"""扫 cover_noise_strength（chinese_female + rock），audio_cover_strength 固定 0.8。"""
from pathlib import Path

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "multi_song_test"
SOURCE = BASE / "sources" / "chinese_female_120s.wav"
LYRICS = (BASE / "03_corrected_lyrics" / "chinese_female.txt").read_text()
OUT = BASE / "sweep_cns_zh"
STYLE = "modern rock, distorted electric guitar, powerful bass, energetic live drums"

CNS_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    handler = AceStepHandler()
    status, success = handler.initialize_service(
        project_root=str(ROOT), config_path="acestep-v15-turbo",
        device="cuda:0", offload_to_cpu=False,
    )
    if not success:
        raise RuntimeError(status)

    for cns in CNS_VALUES:
        name = f"cns{cns:.1f}"
        seed = 20265000
        result = generate_music(
            handler, None,
            params=GenerationParams(
                task_type="cover",
                src_audio=str(SOURCE),
                reference_audio=str(SOURCE),
                caption=(f"{STYLE}, preserve the original female singer and vocal "
                         "identity, preserve the original melody, lyric order, pauses, and structure"),
                lyrics=LYRICS,
                vocal_language="zh",
                duration=120.0,
                audio_cover_strength=0.8,
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
        (OUT / f"{name}.wav").write_bytes(Path(result.audios[0]["path"]).read_bytes())
        logger.info("Saved {}", OUT / f"{name}.wav")


if __name__ == "__main__":
    main()
