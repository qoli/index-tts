export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="${PYTHONPATH:-}:."
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8

uv run python webui.py --model_dir checkpoints --port 7860