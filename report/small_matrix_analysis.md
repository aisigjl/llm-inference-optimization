# Small Matrix Analysis

Scope: Llama-3.1-8B-Instruct, BF16/no quantization, input_len in {128, 512}, output_len=64.

## Validity Check

- Total requests: 288
- Total errors: 0
- All small-matrix benchmark runs are valid.

## Transformers Batch Scaling

| Input Len | Batch Size | Output tok/s | P95 latency (s) | Wall time (s) |
|---:|---:|---:|---:|---:|
| 128 | 1 | 35.80 | 1.897 | 28.60 |
| 128 | 2 | 65.19 | 1.147 | 15.71 |
| 128 | 4 | 121.10 | 0.601 | 8.46 |
| 512 | 1 | 33.70 | 2.026 | 30.39 |
| 512 | 2 | 57.81 | 1.302 | 17.71 |
| 512 | 4 | 92.55 | 0.753 | 11.06 |

## vLLM Concurrency Scaling

| Input Len | Concurrency | Output tok/s | P95 latency (s) | P95 TTFT (s) | Wall time (s) |
|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 40.10 | 1.623 | 0.063 | 51.08 |
| 128 | 4 | 154.26 | 1.632 | 0.052 | 13.28 |
| 128 | 8 | 273.90 | 1.994 | 0.074 | 7.48 |
| 512 | 1 | 39.28 | 1.661 | 0.115 | 52.14 |
| 512 | 4 | 151.81 | 1.652 | 0.058 | 13.49 |
| 512 | 8 | 274.89 | 1.725 | 0.080 | 7.45 |

## Key Findings

- input_len=128: best Transformers batch_size=4 reaches 121.10 output tok/s; best vLLM concurrency=8 reaches 273.90 output tok/s (2.26x).
- input_len=512: best Transformers batch_size=4 reaches 92.55 output tok/s; best vLLM concurrency=8 reaches 274.89 output tok/s (2.97x).
- In this small matrix, vLLM throughput improves strongly from concurrency=1 to 8, while P95 latency only grows moderately.
- Transformers throughput improves with larger batch size, but this is offline batched generation rather than online serving concurrency.
- Peak GPU memory is similar across runs because both paths keep the full model resident; memory differences from KV cache should be tested with longer context and higher concurrency.

## Next Experiments

1. Extend context length to 2048 with output_len=64 for both Transformers and vLLM.
2. Extend vLLM concurrency to 16 and 32 to expose continuous batching behavior and P95 latency growth.
3. Add Transformers INT8 and INT4 runs for input_len=512/output_len=64/batch_size=1 or 2 to measure memory savings.
4. Only after BF16 baseline is stable, add AWQ/GPTQ vLLM runs if local quantized model directories are available.
