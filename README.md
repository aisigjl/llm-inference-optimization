# LLM Inference Optimization: vLLM + Quantization + Benchmark

This project benchmarks local LLM inference on a single 24GB GPU.

It compares:

- HuggingFace Transformers baseline
- vLLM OpenAI-compatible serving
- Transformers INT8 / INT4 loading
- vLLM AWQ / GPTQ / FP8 serving when the local model supports it
- Batch size, context length, concurrency, latency, throughput, and GPU memory


## Quick Start On Current Machine

The default config already points to:

```text
/home/WangChanghui/wch/model/Qwen3-4B-Instruct-2507
```

Run the smallest Transformers smoke benchmark first:

```bash
cd /home/WangChanghui/wch/llm/llm-inference-optimization
conda activate llmInfer

python scripts/run_transformers_baseline.py \
  --input-len 128 \
  --output-len 32 \
  --batch-size 1 \
  --num-requests 4 \
  --quant none \
  --tag smoke_transformers_bf16

python scripts/summarize_results.py
```

Then start vLLM in another terminal:

```bash
cd /home/WangChanghui/wch/llm/llm-inference-optimization
conda activate llmInfer
bash scripts/run_vllm_server.sh
```

Run a small API benchmark:

```bash
python scripts/run_api_benchmark.py \
  --input-len 128 \
  --output-len 32 \
  --concurrency 1 \
  --num-requests 4 \
  --backend vllm \
  --quant none \
  --tag smoke_vllm_bf16

python scripts/summarize_results.py
python scripts/plot_results.py
```

## 1. Create Conda Environment

Use Python 3.10.

```bash
cd /home/WangChanghui/wch/llm/llm-inference-optimization
conda create -n llm-infer-bench python=3.10 -y
conda activate llm-infer-bench
```

Your driver only supports CUDA 12.4, so install PyTorch with the cu124 wheel first:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

Check the environment:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Expected CUDA version from PyTorch should be `12.4`.

## 2. Configure Local Model

Edit:

```text
configs/model.yaml
```

Change:

```yaml
model_path: /home/WangChanghui/wch/model/Qwen3-4B-Instruct-2507
```

For a 24GB GPU, recommended first choices:

- `Qwen3-4B-Instruct`
- `Qwen3-4B-Instruct-AWQ`
- `Qwen3-4B-Instruct-GPTQ-Int4`
- `Llama-3.1-8B-Instruct`

Start with `max_model_len: 4096`. If you hit OOM, try `2048`.

## 3. Run Transformers Baseline

FP16 baseline:

```bash
python scripts/run_transformers_baseline.py \
  --input-len 512 \
  --output-len 128 \
  --batch-size 1 \
  --num-requests 16 \
  --quant none
```

INT8 baseline:

```bash
python scripts/run_transformers_baseline.py \
  --input-len 512 \
  --output-len 128 \
  --batch-size 1 \
  --num-requests 16 \
  --quant int8
```

INT4 baseline with bitsandbytes NF4:

```bash
python scripts/run_transformers_baseline.py \
  --input-len 512 \
  --output-len 128 \
  --batch-size 1 \
  --num-requests 16 \
  --quant int4
```

## 4. Start vLLM Server

In terminal A:

```bash
export MODEL_PATH=/home/WangChanghui/wch/model/Qwen3-4B-Instruct-2507
export SERVED_MODEL_NAME=local-model
export MAX_MODEL_LEN=4096
export DTYPE=bfloat16
bash scripts/run_vllm_server.sh
```

In terminal B:

```bash
python scripts/run_api_benchmark.py \
  --input-len 512 \
  --output-len 128 \
  --concurrency 8 \
  --num-requests 64 \
  --backend vllm \
  --quant none
```

You can also cross-check with vLLM's built-in benchmark:

```bash
MODEL_NAME=local-model INPUT_LEN=512 OUTPUT_LEN=128 NUM_PROMPTS=128 \
  bash scripts/run_vllm_builtin_benchmark.sh
```

## 5. Run AWQ / GPTQ with vLLM

Use an already-quantized local model directory.

AWQ:

```bash
export MODEL_PATH=/home/WangChanghui/wch/model/Qwen3-4B-Instruct-2507-AWQ
export SERVED_MODEL_NAME=local-model
export QUANTIZATION=awq
bash scripts/run_vllm_server.sh
```

Then benchmark:

```bash
python scripts/run_api_benchmark.py \
  --input-len 512 \
  --output-len 128 \
  --concurrency 8 \
  --num-requests 64 \
  --backend vllm \
  --quant awq
```

GPTQ:

```bash
export MODEL_PATH=/home/WangChanghui/wch/model/Qwen3-4B-Instruct-2507-GPTQ-Int4
export SERVED_MODEL_NAME=local-model
export QUANTIZATION=gptq
bash scripts/run_vllm_server.sh
```

Then benchmark with `--quant gptq`.

FP8 depends on your GPU and model support. If your GPU does not support fast FP8, skip it for the first version.

## 6. Run Experiment Matrix

Transformers matrix:

```bash
python scripts/run_matrix.py --mode transformers --quant none
```

vLLM matrix, after the server is running:

```bash
python scripts/run_matrix.py --mode api --backend vllm --quant none
```

You can edit the matrix in:

```text
configs/benchmark.yaml
```

For a 24GB GPU, start conservatively:

```yaml
input_lens: [128, 512, 2048]
output_lens: [64, 128]
batch_sizes: [1, 2, 4]
concurrencies: [1, 4, 8, 16]
num_requests: 64
```

Then try `4096` context and concurrency `32` after the basic run is stable.

## 7. Summarize and Plot

```bash
python scripts/summarize_results.py
python scripts/plot_results.py
```

Outputs:

```text
results/tables/summary.csv
results/figures/output_tokens_per_sec.png
results/figures/latency_p95_sec.png
results/figures/ttft_p95_sec.png
results/figures/peak_gpu_memory_mb.png
```

## 8. Metrics

The scripts record:

- `output_tokens_per_sec`
- `total_tokens_per_sec`
- `latency_mean_sec`
- `latency_p50_sec`
- `latency_p95_sec`
- `latency_p99_sec`
- `ttft_p50_sec`
- `ttft_p95_sec`
- `peak_gpu_memory_mb`
- `mean_gpu_utilization`

Transformers baseline does not use streaming, so TTFT is empty there. vLLM API benchmark uses streaming and records TTFT.

## 9. Common Issues

### vLLM installation fails

This usually means the wheel pulled by pip does not match your CUDA/PyTorch stack. Keep PyTorch at `2.5.1+cu124`, and try:

```bash
pip install "vllm==0.7.3"
```

If it still fails, try:

```bash
pip install "vllm==0.7.2"
```

### Out of memory

Try these in order:

1. Set `max_model_len: 2048` in `configs/model.yaml`.
2. Lower `--concurrency`.
3. Lower `--batch-size`.
4. Use AWQ or GPTQ.
5. Set `GPU_MEMORY_UTILIZATION=0.85` for vLLM.

### AWQ/GPTQ does not start

Make sure the local model is actually an AWQ/GPTQ model. Do not pass `QUANTIZATION=awq` to a normal FP16 model.

## 10. Resume Bullet

```text
Built an LLM inference benchmark platform comparing HuggingFace Transformers and vLLM under FP16, INT8, AWQ, and GPTQ settings. Implemented async API load testing with configurable context length, output length, concurrency, and batch size, collecting tokens/s, TTFT, P50/P95/P99 latency, GPU utilization, and peak memory to analyze deployment tradeoffs on a 24GB single-GPU environment.
```
