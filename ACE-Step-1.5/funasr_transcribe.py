#!/usr/bin/env python3
"""Transcribe singing audio with FunASR SenseVoiceSmall."""

import argparse
import json
import re
from pathlib import Path

from funasr import AutoModel


def clean_text(text: str) -> str:
    """Remove SenseVoice control tags while preserving recognized words."""
    return re.sub(r"<\|[^|]+\|>", "", text).strip()


def main() -> None:
    """Load FunASR once and transcribe all requested audio files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=args.device,
    )

    records = []
    for audio_path in args.audio:
        result = model.generate(
            input=audio_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        text = " ".join(clean_text(item.get("text", "")) for item in result).strip()
        records.append({"path": str(Path(audio_path).resolve()), "text": text, "raw": result})
        print(f"{audio_path}\n{text}\n")

    Path(args.output).write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
