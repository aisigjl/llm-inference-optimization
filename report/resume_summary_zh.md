# 中文简历与面试表述

## 项目名称

大模型推理优化与部署 Benchmark

## 一句话介绍

搭建了一套面向本地大模型部署的推理性能评测框架，系统比较 Transformers、vLLM 在线服务和 INT8/INT4 量化在吞吐、延迟、TTFT、显存和并发扩展性上的表现。

## 简历 Bullet

- 搭建大模型推理优化 benchmark 框架，基于 `Llama-3.1-8B-Instruct` 对比 HuggingFace Transformers、vLLM OpenAI-compatible serving 和 bitsandbytes INT8/INT4 量化方案。
- 设计 batch size、输入长度、输出长度、并发数等实验矩阵，采集 output tokens/s、total tokens/s、P50/P95 latency、TTFT、峰值 GPU 显存和 GPU 利用率等指标。
- 在 512-token 输入场景下，vLLM continuous batching 将输出吞吐从 Transformers 最优 batch baseline 的 `92.55 tok/s` 提升到 concurrency=8 的 `274.89 tok/s`，并在 concurrency=32 达到 `853.30 tok/s`。
- 在 2048-token 长上下文场景下，vLLM concurrency=8 达到 `248.63 output tok/s`，约为 Transformers 长上下文最优 baseline 的 `6.61x`。
- 对比 BF16、INT8、INT4 推理性能，INT4 将峰值显存从 `17.82GB` 降至 `6.83GB`，节省约 `61.7%`，同时保留较高吞吐。
- 排查并修复 GPU 监控中的 CUDA 逻辑 GPU 与 NVML 物理 GPU 编号不一致问题，保证显存和利用率指标可信。

## 面试讲法

这个项目主要是为了证明我理解大模型从训练到落地部署之间的性能问题。我没有只测单次推理速度，而是把评测拆成了离线 batch generation 和在线 serving 两类场景，分别比较 Transformers 和 vLLM 的表现。

我重点关注几个指标：output tokens/s、P95 latency、TTFT、峰值显存和 GPU 利用率。实验结果显示，vLLM 在并发场景下因为 continuous batching 能显著提升吞吐；而 Transformers batch 更适合作为离线批处理 baseline，不适合直接代表在线服务。

在量化实验里，我发现 INT4 能显著降低显存，但 INT8 在当前环境反而变慢。这说明部署优化不能只根据理论判断，必须结合具体硬件、kernel 和推理框架做端到端 benchmark。

这个项目里我还遇到了一个实际工程问题：使用 `CUDA_VISIBLE_DEVICES` 时，PyTorch 看到的 GPU 编号和 NVML 读取的物理 GPU 编号不一致，导致一开始显存监控读错卡。我修复了这个问题，使 benchmark 的 GPU 指标更可靠。

## 面试官可能追问

### 为什么 vLLM 比 Transformers 快？

Transformers `generate` 更像单机离线批处理，在线并发能力弱。vLLM 通过 continuous batching、PagedAttention 和 KV cache 管理，在多个请求同时到来时能更充分利用 GPU，因此并发越高，吞吐优势越明显。

### 为什么 vLLM 显存更高？

vLLM 是服务框架，会根据 `max_model_len` 和 `gpu_memory_utilization` 为 KV cache 预留显存，所以即使请求较短，也可能看到较高显存占用。这是用显存换在线服务吞吐和并发能力。

### 为什么 INT8 比 INT4 还慢？

量化不一定带来速度提升。速度取决于具体 kernel、硬件支持、内存访问模式和框架实现。本实验里 INT8 节省了显存，但吞吐下降明显；INT4 显存收益更大，同时吞吐损失较小。因此量化方案必须实测。

### 如果要线上部署你会怎么选？

如果目标是在线多用户服务，我会优先选 vLLM，并根据 P95 latency 目标调 concurrency、`max_model_len` 和 `gpu_memory_utilization`。如果是低显存单卡部署，或者只是本地离线推理，我会考虑 INT4/NF4 量化。
