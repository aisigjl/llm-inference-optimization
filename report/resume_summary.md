# Resume Summary

## Project Name

LLM Inference Optimization and Serving Benchmark

## One-line Description

Built a reproducible benchmark system for comparing HuggingFace Transformers, vLLM serving, and INT8/INT4 quantization across throughput, latency, TTFT, GPU memory, and concurrency.

## Resume Bullets

- Built an LLM inference benchmark framework for `Llama-3.1-8B-Instruct`, covering Transformers baseline, vLLM OpenAI-compatible serving, and bitsandbytes INT8/INT4 quantization.
- Designed experiments across batch size, input length, output length, and serving concurrency; measured output tokens/s, P50/P95 latency, TTFT, GPU memory, and GPU utilization.
- Demonstrated that vLLM continuous batching improved output throughput from `92.55 tok/s` Transformers best batch baseline to `274.89 tok/s` at concurrency 8 and `853.30 tok/s` at concurrency 32 for 512-token prompts.
- Evaluated long-context serving at 2048-token prompts, where vLLM concurrency 8 achieved `248.63 output tok/s`, about `6.61x` the best Transformers long-context baseline.
- Compared BF16, INT8, and INT4 inference; INT4 reduced peak GPU memory by about `61.7%` versus BF16 while preserving most of the baseline throughput in this setup.
- Found and fixed a GPU monitoring bug caused by `CUDA_VISIBLE_DEVICES` logical GPU IDs not matching NVML physical GPU IDs, improving the reliability of memory and utilization metrics.

## Interview Explanation

This project is meant to show that I understand model deployment beyond training. I did not only run a model once and report speed; I built a benchmark that separates offline batch generation from online serving, measures TTFT and P95 latency, and compares how concurrency, context length, and quantization affect throughput and memory. The main conclusion is that vLLM is much better for concurrent serving because of continuous batching, while quantization is useful for memory reduction but must be benchmarked because it does not always improve speed.
