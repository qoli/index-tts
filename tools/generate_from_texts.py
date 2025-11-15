#!/usr/bin/env python3
"""Batch text-to-speech helper script that mirrors the README usage examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from indextts.infer_v2 import IndexTTS2


WEBUI_EMO_WEIGHT_DEFAULT = 0.65
WEBUI_GENERATION_DEFAULTS = {
    "do_sample": True,
    "top_p": 0.8,
    "top_k": 30,
    "temperature": 0.8,
    "length_penalty": 0.0,
    "num_beams": 3,
    "repetition_penalty": 10.0,
    "max_mel_tokens": 1500,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate speech with IndexTTS2 using a text file and a reference voice clip. "
            "Defaults mirror the WebUI panels so CLI results stay consistent."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default=WEBUI_EMO_WEIGHT_DEFAULT,
        help="Emotion weight slider value used by the WebUI (default: 0.65)",
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
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=WEBUI_GENERATION_DEFAULTS["do_sample"],
        help="Match the WebUI do_sample toggle (use --no-do-sample to disable)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=WEBUI_GENERATION_DEFAULTS["temperature"],
        help="Softmax temperature applied during generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=WEBUI_GENERATION_DEFAULTS["top_p"],
        help="Top-p nucleus sampling probability mass",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=WEBUI_GENERATION_DEFAULTS["top_k"],
        help="Top-k cutoff for the autoregressive GPT (<=0 disables the clamp)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=WEBUI_GENERATION_DEFAULTS["num_beams"],
        help="Beam search width used for deterministic decoding",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=WEBUI_GENERATION_DEFAULTS["length_penalty"],
        help="Length penalty passed to the autoregressive decoder",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=WEBUI_GENERATION_DEFAULTS["repetition_penalty"],
        help="Penalty to discourage phrase repetitions",
    )
    parser.add_argument(
        "--max-mel-tokens",
        type=int,
        default=WEBUI_GENERATION_DEFAULTS["max_mel_tokens"],
        help="Maximum mel tokens (lower values truncate audio sooner)",
    )
    parser.add_argument(
        "--interval-silence",
        type=int,
        default=0,
        help="Silence between segments in milliseconds when the model splits long texts (default: 0)",
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
        "--use-cuda-kernel",
        action="store_true",
        help="Enable the CUDA inference kernels exposed by the WebUI",
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
        use_cuda_kernel=args.use_cuda_kernel,
        use_torch_compile=args.use_torch_compile,
    )

    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    generation_kwargs = {
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "top_k": top_k,
        "temperature": args.temperature,
        "length_penalty": args.length_penalty,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "max_mel_tokens": args.max_mel_tokens,
    }

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
        **generation_kwargs,
    )


if __name__ == "__main__":
    main()
