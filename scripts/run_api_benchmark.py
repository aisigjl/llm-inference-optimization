#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm_asyncio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark.datasets import build_prompts
from src.benchmark.metrics import RequestMetric, summarize_request_metrics, write_json, write_jsonl
from src.clients.openai_client import OpenAICompatClient
from src.utils.config import ensure_dir, load_yaml
from src.utils.gpu_monitor import GpuMonitor
from src.utils.token_counter import count_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--bench-config", default="configs/benchmark.yaml")
    parser.add_argument("--serving-config", default="configs/serving.yaml")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--quant", default="none")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


async def bounded_request(
    request_id: int,
    prompt: str,
    client: OpenAICompatClient,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    model_path: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> RequestMetric:
    async with semaphore:
        input_tokens = count_tokens(prompt, model_path)
        try:
            result = await client.generate(
                session=session,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )
            output_tokens = count_tokens(result["text"], model_path)
            return RequestMetric(
                request_id=request_id,
                ok=True,
                latency_sec=float(result["latency_sec"]),
                ttft_sec=result["ttft_sec"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            return RequestMetric(
                request_id=request_id,
                ok=False,
                latency_sec=0.0,
                ttft_sec=None,
                input_tokens=input_tokens,
                output_tokens=0,
                error=str(exc),
            )


async def run() -> None:
    args = parse_args()
    model_cfg = load_yaml(ROOT / args.model_config)
    bench_cfg = load_yaml(ROOT / args.bench_config)
    serving_cfg = load_yaml(ROOT / args.serving_config)

    model_path = model_cfg["model_path"]
    served_model_name = serving_cfg["vllm"].get("served_model_name") or model_cfg.get("model_name", "local-model")
    api_cfg = serving_cfg["api"]
    max_tokens = args.output_len or int(bench_cfg["generation"]["max_new_tokens"])
    num_requests = args.num_requests or int(bench_cfg["matrix"]["num_requests"])
    output_dir = ensure_dir(ROOT / bench_cfg["output"]["raw_dir"])
    prompts = build_prompts(bench_cfg["dataset"]["prompts"], args.input_len, num_requests)

    client = OpenAICompatClient(
        base_url=api_cfg["base_url"],
        model=served_model_name,
        endpoint=api_cfg.get("endpoint", "chat"),
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    temperature = float(bench_cfg["generation"].get("temperature", 0.0))
    top_p = float(bench_cfg["generation"].get("top_p", 1.0))

    with GpuMonitor(args.gpu_index) as monitor:
        async with aiohttp.ClientSession() as session:
            tasks = [
                bounded_request(
                    i,
                    prompt,
                    client,
                    session,
                    semaphore,
                    model_path,
                    max_tokens,
                    temperature,
                    top_p,
                )
                for i, prompt in enumerate(prompts)
            ]
            records = await tqdm_asyncio.gather(*tasks, desc="API benchmark")

    tag = args.tag or f"{args.backend}_{args.quant}_in{args.input_len}_out{max_tokens}_c{args.concurrency}"
    summary = summarize_request_metrics(
        records,
        {
            "backend": args.backend,
            "model": model_cfg.get("model_name", Path(model_path).name),
            "quant": args.quant,
            "input_len_target": args.input_len,
            "output_len_target": max_tokens,
            "concurrency": args.concurrency,
            "peak_gpu_memory_mb": monitor.peak_used_mb,
            "mean_gpu_utilization": monitor.mean_gpu_utilization,
        },
    )
    write_jsonl(output_dir / f"{tag}.jsonl", records)
    write_json(output_dir / f"{tag}.summary.json", summary)
    print(f"Wrote {output_dir / f'{tag}.summary.json'}")


if __name__ == "__main__":
    asyncio.run(run())
