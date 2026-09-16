# Smoke Benchmark Comparison

| Backend | Success | Output tok/s | Total tok/s | Mean latency | P95 latency | P95 TTFT | Peak GPU MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| transformers | 4/4 | 31.7628 | 163.5290 | 0.9868 | 1.2561 | - | 17820.2500 |
| vllm | 4/4 | 34.0678 | 174.3311 | 0.8118 | 0.8340 | 0.0837 | 17820.2500 |

- vLLM output throughput is 1.07x of Transformers in this tiny concurrency=1 smoke run.
- vLLM P95 latency is 0.66x of Transformers in this run.
- This is only a smoke sanity check; the real vLLM advantage should be measured under higher concurrency.
