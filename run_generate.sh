#!/usr/bin/env bash
# Helper wrapper that runs the generate_from_texts.py script inside the uv environment.
# Mirrors the README guidance: run scripts via `uv run <file.py>` and ensure PYTHONPATH includes repo root.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="${PYTHONPATH:-}:."
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8

echo ">> Running generate_from_texts.py via uv..."
uv run python tools/generate_from_texts.py \
  --voice-reference examples/voice_11.wav \
  --text-file input_texts.txt \
  --output outputs/input_texts.wav \
  "$@"


# uv run python tools/generate_from_texts.py \
#   --voice-reference voiceReference.mp3 \
#   --text-file input_texts.txt \
#   --output outputs/input_texts.wav \
#   "$@"
