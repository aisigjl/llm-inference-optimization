# Implementation Plan

## Goal

Build an LLM inference optimization benchmark that compares Transformers baseline, vLLM serving, and quantized inference under controlled workloads.

## Phase 1: Smoke Run

1. Validate environment and CUDA.
2. Run a tiny Transformers benchmark: input 128, output 32, batch 1, requests 4.
3. Summarize results and confirm raw JSONL/summary JSON are generated.

## Phase 2: Transformers Baseline

Run BF16 baseline across:

- input length: 128, 512, 2048
- output length: 64, 128
- batch size: 1, 2, 4

Metrics:

- latency p50/p95/p99
- output tokens/s
- total tokens/s
- peak GPU memory
- GPU utilization

## Phase 3: vLLM Serving

Start vLLM OpenAI-compatible server and run API benchmark across:

- input length: 128, 512, 2048
- output length: 64, 128
- concurrency: 1, 4, 8, 16, 32

Key comparison:

- vLLM vs Transformers under low concurrency
- vLLM throughput scaling under higher concurrency
- P95 latency growth under high concurrency

## Phase 4: Quantization

Start with practical first version:

- Transformers INT8 via bitsandbytes
- Transformers INT4 via bitsandbytes NF4
- vLLM AWQ/GPTQ only if local quantized model directories are available

Do not block the project on producing quantized weights from scratch. Existing AWQ/GPTQ model directories are acceptable.

## Phase 5: Report

Generate:

- `results/tables/summary.csv`
- figures under `results/figures/`
- final `report/benchmark_report.md`

The report should explain tradeoffs rather than only list numbers.
