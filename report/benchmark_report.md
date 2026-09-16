# LLM Inference Optimization Benchmark Report

## Executive Summary

This project benchmarks `Llama-3.1-8B-Instruct` inference across HuggingFace Transformers, vLLM serving, and bitsandbytes INT8/INT4 loading. The benchmark measures output throughput, total throughput, P50/P95 latency, TTFT, GPU memory, and GPU utilization under different input lengths, batch sizes, and serving concurrency.

All benchmark runs in `results/tables/summary.csv` completed successfully with `num_error=0`.

## Environment

- Model: `/home/WangChanghui/wch/model/Llama-3.1-8B-Instruct`
- Model name: `llama3-1-8b-instruct`
- Precision baseline: BF16 / no quantization
- Transformers backend: HuggingFace `AutoModelForCausalLM.generate`
- Serving backend: vLLM OpenAI-compatible API
- Quantization tested: bitsandbytes INT8 and NF4 INT4 for Transformers
- Main result table: `results/tables/summary.csv`
- Figures: `results/figures/`

## Result Validity

- Total benchmark variants: 23 summary files, including smoke tests.
- Main matrix variants: Transformers BF16 batch tests, vLLM concurrency tests, long-context tests, and INT8/INT4 tests.
- Error count: all runs have `num_error=0`.
- GPU monitoring issue was found and fixed: `src/utils/gpu_monitor.py` now maps `CUDA_VISIBLE_DEVICES` logical GPU IDs to NVML physical GPU IDs.

## Key Results

### 1. vLLM improves online serving throughput

For `input_len=512`, `output_len=64`:

| Backend | Config | Output tok/s | P95 latency | Peak GPU memory |
|---|---:|---:|---:|---:|
| Transformers BF16 | batch_size=1 | 33.70 | 2.026s | 17.82GB |
| Transformers BF16 | batch_size=4 | 92.55 | 0.753s | 17.82GB |
| vLLM BF16 | concurrency=1 | 39.28 | 1.661s | 17.82GB |
| vLLM BF16 | concurrency=8 | 274.89 | 1.725s | 17.82GB |
| vLLM BF16 | concurrency=32 | 853.30 | 2.183s | 44.31GB |

Findings:

- vLLM scales strongly with concurrency because continuous batching keeps the GPU occupied.
- At `input_len=512`, vLLM `concurrency=8` reaches `274.89 output tok/s`, about `2.97x` the best Transformers batch result in this matrix.
- At `concurrency=32`, vLLM reaches `853.30 output tok/s`, about `9.22x` the best Transformers batch result, while P95 latency remains around `2.18s`.

### 2. Long context mainly increases TTFT and KV-cache pressure

For `input_len=2048`, `output_len=64`:

| Backend | Config | Output tok/s | P95 latency | P95 TTFT | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| Transformers BF16 | batch_size=1 | 26.71 | 2.580s | - | 17.14GB |
| Transformers BF16 | batch_size=2 | 37.61 | 1.952s | - | 17.92GB |
| vLLM BF16 | concurrency=1 | 33.41 | 1.992s | 0.428s | 43.99GB |
| vLLM BF16 | concurrency=4 | 141.38 | 1.771s | 0.082s | 43.99GB |
| vLLM BF16 | concurrency=8 | 248.63 | 1.913s | 0.112s | 43.99GB |

Findings:

- vLLM still scales under 2048-token prompts: `concurrency=8` reaches `248.63 output tok/s`.
- Compared with the best Transformers long-context result, vLLM `concurrency=8` is about `6.61x` faster in output throughput.
- Long context increases prefill cost and TTFT. This is visible in vLLM `concurrency=1`, where P95 TTFT rises to `0.428s`.
- vLLM memory is higher because the server reserves memory for KV cache according to `max_model_len` and `gpu_memory_utilization`.

### 3. Quantization reduces memory, but speed depends on the kernel path

For Transformers, `input_len=512`, `output_len=64`, `batch_size=1`:

| Quant | Output tok/s | P95 latency | Peak GPU memory | Memory saved vs BF16 |
|---|---:|---:|---:|---:|
| BF16 | 33.70 | 2.026s | 17.82GB | - |
| INT8 | 15.94 | 4.318s | 9.87GB | 44.6% |
| INT4 | 29.75 | 2.370s | 6.83GB | 61.7% |

Findings:

- INT4 gives the best memory reduction in this setup: about `61.7%` less memory than BF16.
- INT4 throughput is close to BF16 in this test, with about `11.7%` lower output tok/s.
- INT8 saves memory but is much slower here, likely due to the specific bitsandbytes kernel path and hardware behavior. This is a useful deployment lesson: quantization should be benchmarked, not assumed to be faster.

## Deployment Interpretation

- For maximum online throughput, prefer vLLM serving with tuned concurrency.
- For memory-constrained single-GPU local inference, INT4 loading is useful when vLLM KV-cache reservation is too large.
- For long-context online serving, control `max_model_len`, `gpu_memory_utilization`, and concurrency together. Higher concurrency improves throughput but increases KV-cache pressure.
- Transformers batch generation is a good offline baseline, but it does not represent realistic multi-user serving behavior.

## Recommended Configuration

For this machine and model:

- Online serving baseline: vLLM BF16, `max_model_len=4096`, concurrency around `8-32` depending on latency target.
- Conservative latency target: vLLM concurrency `8`.
- Maximum throughput target in current matrix: vLLM concurrency `32` for 512-token prompts.
- Memory-constrained fallback: Transformers INT4 / NF4 when vLLM memory reservation is too high.

## Interview Talking Points

- I built a reproducible inference benchmark that compares offline generation and online serving instead of only reporting a single latency number.
- I measured output tokens/s, total tokens/s, P50/P95 latency, TTFT, GPU memory, and GPU utilization.
- I found and fixed a GPU monitoring bug caused by `CUDA_VISIBLE_DEVICES` and NVML physical GPU ID mismatch.
- The benchmark shows vLLM continuous batching gives much better throughput under concurrency, especially compared with plain Transformers generation.
- Quantization reduced memory substantially, but INT8 was slower in this environment, showing why deployment optimization must be measured end to end.

## Artifacts

- Raw results: `results/raw/*.jsonl` and `results/raw/*.summary.json`
- Summary table: `results/tables/summary.csv`
- Figures: `results/figures/*.png`
- Benchmark scripts: `scripts/run_transformers_baseline.py`, `scripts/run_api_benchmark.py`, `scripts/run_matrix.py`
- vLLM server script: `scripts/run_vllm_server.sh`
