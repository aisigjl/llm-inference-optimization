# 大模型推理优化与部署 Benchmark 报告

## 项目概述

本项目围绕 `Llama-3.1-8B-Instruct` 搭建了一套本地大模型推理性能评测框架，用于比较 HuggingFace Transformers、vLLM 在线服务、以及 INT8/INT4 量化方案在不同输入长度、batch size、并发数下的性能表现。

评测指标包括：

- 输出吞吐：output tokens/s
- 总吞吐：total tokens/s
- 延迟：P50 / P95 / P99 latency
- 首 token 延迟：TTFT
- 峰值 GPU 显存
- GPU 平均利用率
- 请求成功率 / 失败数

所有实验结果保存在 `results/tables/summary.csv`，图表保存在 `results/figures/`。

## 实验环境

- 模型：`/home/WangChanghui/wch/model/Llama-3.1-8B-Instruct`
- 模型名称：`llama3-1-8b-instruct`
- 基线精度：BF16
- 离线推理基线：HuggingFace Transformers `generate`
- 在线服务框架：vLLM OpenAI-compatible API
- 量化方案：bitsandbytes INT8 / NF4 INT4
- 结果汇总：`results/tables/summary.csv`

## 数据有效性

本轮 benchmark 的所有 summary 结果中，`num_error=0`，说明请求均成功完成。

实验过程中发现过一个 GPU 监控问题：当使用 `CUDA_VISIBLE_DEVICES` 指定 GPU 时，PyTorch 看到的是逻辑 GPU 编号，而 NVML 读取的是物理 GPU 编号，二者不一致会导致显存和利用率读错卡。项目中已修复该问题：`src/utils/gpu_monitor.py` 会自动将 CUDA 逻辑 GPU 映射到 NVML 物理 GPU。

## 核心实验结论

### 1. vLLM 在在线并发场景下显著提升吞吐

在 `input_len=512`、`output_len=64` 场景下：

| 方案 | 配置 | 输出吞吐 | P95 延迟 | 峰值显存 |
|---|---:|---:|---:|---:|
| Transformers BF16 | batch_size=1 | 33.70 tok/s | 2.026s | 17.82GB |
| Transformers BF16 | batch_size=4 | 92.55 tok/s | 0.753s | 17.82GB |
| vLLM BF16 | concurrency=1 | 39.28 tok/s | 1.661s | 17.82GB |
| vLLM BF16 | concurrency=8 | 274.89 tok/s | 1.725s | 17.82GB |
| vLLM BF16 | concurrency=32 | 853.30 tok/s | 2.183s | 44.31GB |

结论：

- Transformers batch 更像离线批处理基线，不适合直接代表在线多用户服务。
- vLLM 通过 continuous batching 在并发场景下显著提升 GPU 利用率和吞吐。
- 在 512-token 输入场景下，vLLM concurrency=8 的吞吐约为 Transformers 最优 batch baseline 的 `2.97x`。
- vLLM concurrency=32 的吞吐达到 `853.30 output tok/s`，约为 Transformers 最优 batch baseline 的 `9.22x`，但显存上升到约 `44.31GB`。

### 2. 长上下文主要增加 prefill 成本和 KV cache 压力

在 `input_len=2048`、`output_len=64` 场景下：

| 方案 | 配置 | 输出吞吐 | P95 延迟 | P95 TTFT | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| Transformers BF16 | batch_size=1 | 26.71 tok/s | 2.580s | - | 17.14GB |
| Transformers BF16 | batch_size=2 | 37.61 tok/s | 1.952s | - | 17.92GB |
| vLLM BF16 | concurrency=1 | 33.41 tok/s | 1.992s | 0.428s | 43.99GB |
| vLLM BF16 | concurrency=4 | 141.38 tok/s | 1.771s | 0.082s | 43.99GB |
| vLLM BF16 | concurrency=8 | 248.63 tok/s | 1.913s | 0.112s | 43.99GB |

结论：

- 长上下文会增加 prefill 阶段成本，因此 TTFT 更敏感。
- vLLM 在 2048-token 输入下依然能通过并发提升吞吐。
- vLLM concurrency=8 的输出吞吐为 `248.63 tok/s`，约为 Transformers 长上下文最优结果的 `6.61x`。
- vLLM 显存较高，主要来自服务端为 KV cache 预留显存，这与 `max_model_len` 和 `gpu_memory_utilization` 相关。

### 3. 量化能明显降低显存，但不一定提升速度

在 Transformers `input_len=512`、`output_len=64`、`batch_size=1` 场景下：

| 精度 / 量化 | 输出吞吐 | P95 延迟 | 峰值显存 | 相比 BF16 显存节省 |
|---|---:|---:|---:|---:|
| BF16 | 33.70 tok/s | 2.026s | 17.82GB | - |
| INT8 | 15.94 tok/s | 4.318s | 9.87GB | 44.6% |
| INT4 | 29.75 tok/s | 2.370s | 6.83GB | 61.7% |

结论：

- INT4 显存收益最好，峰值显存从 BF16 的 `17.82GB` 降到 `6.83GB`，节省约 `61.7%`。
- INT4 吞吐接近 BF16，本实验中输出吞吐下降约 `11.7%`。
- INT8 虽然也节省显存，但速度明显变慢，说明量化不能只看位宽，需要结合具体 kernel、硬件和推理框架做实际 benchmark。

## 工程结论

- 如果目标是在线服务吞吐，优先使用 vLLM，而不是直接用 Transformers `generate` 提供服务。
- 如果目标是单卡低显存部署，可以优先尝试 INT4/NF4 量化。
- 如果需要长上下文服务，需要同时调节 `max_model_len`、并发数和 KV cache 显存预留，否则容易出现显存压力。
- 评估推理优化时不能只看平均延迟，需要同时看 P95 latency、TTFT、tokens/s 和显存。

## 推荐部署策略

对于当前模型和机器环境：

- 在线服务推荐：vLLM BF16，`max_model_len=4096`。
- 稳定延迟优先：concurrency 选择 `8` 左右。
- 吞吐优先：512-token 输入场景下可尝试 concurrency `32`。
- 显存受限：使用 Transformers INT4/NF4 作为 fallback。

## 项目价值

这个项目证明的不只是“会跑模型”，而是理解模型落地时的真实约束：吞吐、延迟、首 token 延迟、KV cache、并发、显存和量化收益之间存在 trade-off。项目中还包含可复现实验脚本、结果表格、图表和最终报告，适合作为大模型算法 / 推理优化 / 模型部署方向的项目经历。
