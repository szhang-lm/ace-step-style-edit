"""Run the controlled Chinese-female ACE-Step cover case on Apple MPS."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler


REPO_ROOT = Path(__file__).resolve().parent.parent
THESIS_ROOT = REPO_ROOT.parent
TEST_KIT = THESIS_ROOT / "experiments" / "cover_test_kit"

STYLES = {
    "jazz": "smooth jazz, upright piano, brushed drums, double bass, warm saxophone",
    "rock": "modern rock, distorted electric guitar, powerful bass, energetic live drums",
    "electronic": (
        "modern electronic pop, synthesizers, electronic drums, deep bass, "
        "polished production"
    ),
    "folk": (
        "intimate acoustic folk, acoustic guitar, light percussion, "
        "natural warm arrangement"
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse experiment settings from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=sorted(STYLES), default="jazz")
    parser.add_argument("--cover-noise-strength", type=float, default=0.6)
    parser.add_argument("--audio-cover-strength", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20264000)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--vocal-mask", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="override the cover source audio (e.g. a separated vocal stem)",
    )
    parser.add_argument(
        "--lyrics-file", type=Path, default=None,
        help="override the lyrics text file",
    )
    parser.add_argument("--vocal-cns", type=float, default=0.6)
    parser.add_argument("--mask-fade-seconds", type=float, default=0.25)
    parser.add_argument(
        "--source-as-reference",
        action="store_true",
        help="Use the source song as the acoustic reference, matching the archived baseline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THESIS_ROOT / "experiments" / "nonuniform_noise" / "outputs",
    )
    return parser.parse_args()


def main() -> None:
    """Initialize ACE-Step without the LM and generate one controlled cover."""
    args = parse_args()
    source = args.source or (TEST_KIT / "sources" / "chinese_female_120s.wav")
    lyrics_path = args.lyrics_file or (
        TEST_KIT / "03_corrected_lyrics" / "chinese_female.txt"
    )
    if not source.is_file() or not lyrics_path.is_file():
        raise FileNotFoundError(f"Missing test-kit input under {TEST_KIT}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.vocal_mask is not None:
        os.environ["ACESTEP_VOCAL_MASK_PATH"] = str(args.vocal_mask.resolve())
        os.environ["ACESTEP_VOCAL_CNS"] = str(args.vocal_cns)
        os.environ["ACESTEP_VOCAL_MASK_FADE_SEC"] = str(args.mask_fade_seconds)
    lyrics = lyrics_path.read_text(encoding="utf-8").strip()
    caption = (
        f"{STYLES[args.style]}, preserve the original female singer and vocal identity, "
        "preserve the original melody, lyric order, pauses, and structure"
    )

    logger.info("Initializing ACE-Step DiT on MPS (LM disabled, MLX DiT disabled)")
    init_started = time.time()
    dit_handler = AceStepHandler()
    status, initialized = dit_handler.initialize_service(
        project_root=str(REPO_ROOT),
        config_path="acestep-v15-turbo",
        device="mps",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
        use_mlx_dit=False,
    )
    if not initialized:
        raise RuntimeError(status)
    logger.info(f"Model initialized in {time.time() - init_started:.1f}s\n{status}")

    params = GenerationParams(
        task_type="cover",
        src_audio=str(source),
        reference_audio=str(source) if args.source_as_reference else None,
        caption=caption,
        lyrics=lyrics,
        instrumental=False,
        vocal_language="zh",
        duration=None,
        audio_cover_strength=args.audio_cover_strength,
        cover_noise_strength=args.cover_noise_strength,
        inference_steps=args.steps,
        guidance_scale=1.0,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        seed=args.seed,
    )
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=False,
        seeds=[args.seed],
        audio_format="wav",
    )

    logger.info(
        f"Generating style={args.style}, cover_noise_strength="
        f"{args.cover_noise_strength}, seed={args.seed}"
    )
    generation_started = time.time()
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=LLMHandler(),
        params=params,
        config=config,
        save_dir=str(args.output_dir),
    )
    elapsed = time.time() - generation_started
    if not result.success:
        raise RuntimeError(result.error or result.status_message)

    manifest = {
        "style": args.style,
        "cover_noise_strength": args.cover_noise_strength,
        "audio_cover_strength": args.audio_cover_strength,
        "seed": args.seed,
        "steps": args.steps,
        "vocal_mask": str(args.vocal_mask) if args.vocal_mask else None,
        "vocal_cns": args.vocal_cns if args.vocal_mask else None,
        "mask_fade_seconds": args.mask_fade_seconds if args.vocal_mask else None,
        "source": str(source),
        "reference_audio": str(source) if args.source_as_reference else None,
        "lyrics": str(lyrics_path),
        "caption": caption,
        "elapsed_seconds": elapsed,
        "audios": result.audios,
        "status_message": result.status_message,
    }
    manifest_path = args.output_dir / (
        f"manifest_{args.style}_cns{args.cover_noise_strength:.2f}_seed{args.seed}.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"Generation completed in {elapsed:.1f}s")
    logger.info(f"Manifest: {manifest_path}")
    for audio in result.audios:
        logger.info(f"Audio: {audio.get('path', audio)}")


if __name__ == "__main__":
    main()
