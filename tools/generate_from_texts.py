#!/usr/bin/env python3
"""Batch text-to-speech helper script that mirrors the README usage examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from indextts.infer_v2 import IndexTTS2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate speech with IndexTTS2 using a text file and a reference voice clip. "
            "Defaults follow the README quickstart examples."
        )
    )
    parser.add_argument(
        "--cfg-path",
        default="checkpoints/config.yaml",
        help="Path to the model configuration file (default: checkpoints/config.yaml)",
    )
    parser.add_argument(
        "--model-dir",
        default="checkpoints",
        help="Directory that holds the downloaded checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--voice-reference",
        required=True,
        help="Reference audio used for cloning the target voice (spk_audio_prompt)",
    )
    parser.add_argument(
        "--text-file",
        required=True,
        help="UTF-8 text file that contains the narration to synthesize",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path of the WAV/FLAC/MP3 file that will store the generated speech",
    )
    parser.add_argument(
        "--emo-audio",
        help="Optional emotion reference audio clip; defaults to the voice reference when omitted",
    )
    parser.add_argument(
        "--emo-alpha",
        type=float,
        default=1.0,
        help="Blending factor for emotional prompts (default: 1.0)",
    )
    parser.add_argument(
        "--emo-vector",
        help=(
            "Comma separated or JSON list of 8 floats describing [happy, angry, sad, afraid, "
            "disgusted, melancholic, surprised, calm]"
        ),
    )
    parser.add_argument(
        "--emo-text",
        help="Optional free-form text that describes the emotion style when --use-emo-text is enabled",
    )
    parser.add_argument(
        "--use-emo-text",
        action="store_true",
        help="Derive the emotional vector automatically from either --emo-text or the narration text",
    )
    parser.add_argument(
        "--use-random",
        action="store_true",
        help="Enable stochastic decoding for more diverse audio (default: disabled)",
    )
    parser.add_argument(
        "--interval-silence",
        type=int,
        default=200,
        help="Silence between segments in milliseconds when the model splits long texts (default: 200)",
    )
    parser.add_argument(
        "--max-text-tokens",
        type=int,
        default=120,
        help="Maximum number of text tokens per segment before automatic splitting (default: 120)",
    )
    parser.add_argument(
        "--more-segment-before",
        type=int,
        default=0,
        help="Internal override for pre-padding segments; leave at 0 unless you know what you're doing",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Force float16 inference (recommended on supported GPUs)",
    )
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        help="Run the autoregressive GPT under DeepSpeed inference",
    )
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Enable torch.compile for the s2mel model (experimental)",
    )
    parser.add_argument(
        "--device",
        help="Override the target device string (e.g., cuda:0, cpu, mps)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the internal inference arguments for debugging",
    )
    return parser.parse_args()


def _parse_emo_vector(raw_value: str | None) -> Sequence[float] | None:
    if not raw_value:
        return None

    cleaned = raw_value.strip()
    if not cleaned:
        return None

    if cleaned.startswith("["):
        values = json.loads(cleaned)
    else:
        values = [float(chunk) for chunk in cleaned.split(",") if chunk.strip()]

    if len(values) != 8:
        raise ValueError(
            f"Expected 8 emotion values, but received {len(values)} entries: {values}"
        )
    return values


def _load_text(text_path: Path) -> str:
    data = text_path.read_text(encoding="utf-8")
    # Keep intentional newlines while trimming leading/trailing whitespace.
    return data.strip()


def main() -> None:
    args = parse_args()
    text_path = Path(args.text_file)
    if not text_path.is_file():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    voice_reference = Path(args.voice_reference)
    if not voice_reference.is_file():
        raise FileNotFoundError(f"Voice reference not found: {voice_reference}")

    emo_audio = Path(args.emo_audio) if args.emo_audio else None

    narration = _load_text(text_path)
    if not narration:
        raise ValueError("The provided text file is empty after trimming whitespace.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    emo_vector = _parse_emo_vector(args.emo_vector)

    tts = IndexTTS2(
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        use_fp16=args.use_fp16,
        device=args.device,
        use_deepspeed=args.use_deepspeed,
        use_torch_compile=args.use_torch_compile,
    )

    tts.infer(
        spk_audio_prompt=str(voice_reference),
        text=narration,
        output_path=str(output_path),
        emo_audio_prompt=str(emo_audio) if emo_audio else None,
        emo_alpha=args.emo_alpha,
        emo_vector=emo_vector,
        use_emo_text=args.use_emo_text,
        emo_text=args.emo_text,
        use_random=args.use_random,
        interval_silence=args.interval_silence,
        verbose=args.verbose,
        max_text_tokens_per_segment=args.max_text_tokens,
        more_segment_before=args.more_segment_before,
    )


if __name__ == "__main__":
    main()
