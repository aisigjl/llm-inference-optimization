#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/absolute/path/to/your/local/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-model}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
QUANTIZATION="${QUANTIZATION:-}"

args=(
  "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --trust-remote-code
  --enable-prefix-caching
  --disable-log-requests
)

if [[ -n "$QUANTIZATION" && "$QUANTIZATION" != "none" ]]; then
  args+=(--quantization "$QUANTIZATION")
fi

vllm serve "${args[@]}"
