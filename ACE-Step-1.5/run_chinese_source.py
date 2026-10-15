#!/usr/bin/env python3
"""Generate a controlled Chinese source song with fully known lyrics."""

from pathlib import Path

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "multi_song_test" / "sources"
LYRICS = """[Verse 1]
清晨的风吹过安静街口
阳光落在熟悉的窗口
我沿着昨天留下的脚步
寻找心中没有说完的路

[Pre-Chorus]
远方的云慢慢向前走
像我们曾经许下的以后

[Chorus]
让这首歌陪我走过山海
每一句话都不会被掩埋
无论时间把故事怎样更改
我还记得你眼里的期待

[Verse 2]
夜晚的灯照亮旧的站台
离别的人还没有回来
我把思念写进温柔节拍
等下一场花在春天盛开

[Pre-Chorus]
远方的云慢慢向前走
像我们曾经许下的以后

[Chorus]
让这首歌陪我走过山海
每一句话都不会被掩埋
无论时间把故事怎样更改
我还记得你眼里的期待

[Bridge]
如果风雨遮住天空色彩
就让心跳成为我的节拍
走过漫长黑夜不再徘徊
黎明终会为我们而来

[Final Chorus]
让这首歌陪我走过山海
每一句话都不会被掩埋
无论时间把故事怎样更改
我还记得你眼里的期待
让这旋律穿过人潮大海
告诉未来我们依然存在

[Outro]
我们依然存在
"""


def main() -> None:
    """Generate and save one deterministic 120-second Mandarin pop song."""
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

    result = generate_music(
        handler,
        None,
        params=GenerationParams(
            task_type="text2music",
            caption=(
                "modern Mandarin pop ballad, clear prominent female lead vocal, warm piano, "
                "acoustic guitar, soft drums, emotional but natural singing"
            ),
            lyrics=LYRICS,
            vocal_language="zh",
            duration=120.0,
            bpm=88,
            keyscale="C major",
            timesignature="4",
            inference_steps=8,
            guidance_scale=1.0,
            seed=20261201,
            thinking=False,
        ),
        config=GenerationConfig(
            batch_size=1,
            audio_format="wav",
            use_random_seed=False,
            seeds=[20261201],
        ),
        save_dir=str(OUT / "chinese_generated_raw"),
    )
    if not result.success:
        raise RuntimeError(result.status_message)
    generated = Path(result.audios[0]["path"])
    final_path = OUT / "chinese_female_120s.wav"
    final_path.write_bytes(generated.read_bytes())
    (OUT / "chinese_female_lyrics.txt").write_text(LYRICS, encoding="utf-8")
    logger.info("Saved {}", final_path)


if __name__ == "__main__":
    main()
