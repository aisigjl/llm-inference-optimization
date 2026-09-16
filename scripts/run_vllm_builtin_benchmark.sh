#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-local-model}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
REQUEST_RATE="${REQUEST_RATE:-inf}"

vllm bench serve \
  --backend openai-chat \
  --model "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --request-rate "$REQUEST_RATE"

