#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark.datasets import build_prompts
from src.benchmark.metrics import RequestMetric, summarize_request_metrics, write_json, write_jsonl
from src.utils.config import ensure_dir, load_yaml
from src.utils.gpu_monitor import GpuMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--bench-config", default="configs/benchmark.yaml")
    parser.add_argument("--quant", choices=["none", "int8", "int4"], default="none")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def load_model(model_path: str, dtype: str, quant: str, trust_remote_code: bool):
    torch_dtype = torch.float16 if dtype in {"float16", "fp16"} else torch.bfloat16
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
        "torch_dtype": torch_dtype,
    }
    if quant == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs.pop("torch_dtype", None)
    elif quant == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs.pop("torch_dtype", None)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return tokenizer, model


def main() -> None:
    args = parse_args()
    model_cfg = load_yaml(ROOT / args.model_config)
    bench_cfg = load_yaml(ROOT / args.bench_config)

    model_path = model_cfg["model_path"]
    model_name = model_cfg.get("model_name", Path(model_path).name)
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    max_new_tokens = args.output_len or int(bench_cfg["generation"]["max_new_tokens"])
    output_dir = ensure_dir(ROOT / bench_cfg["output"]["raw_dir"])

    tokenizer, model = load_model(
        model_path=model_path,
        dtype=model_cfg.get("dtype", "float16"),
        quant=args.quant,
        trust_remote_code=trust_remote_code,
    )

    base_prompts = bench_cfg["dataset"]["prompts"]
    prompts = build_prompts(base_prompts, args.input_len, args.num_requests)
    records: list[RequestMetric] = []

    with GpuMonitor(args.gpu_index) as monitor:
        for start_idx in tqdm(range(0, len(prompts), args.batch_size), desc="Transformers baseline"):
            batch_prompts = prompts[start_idx : start_idx + args.batch_size]
            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            input_counts = encoded["attention_mask"].sum(dim=1).tolist()
            start = time.perf_counter()
            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                elapsed = time.perf_counter() - start
                per_request_latency = elapsed / len(batch_prompts)
                for offset, output in enumerate(outputs):
                    input_tokens = int(input_counts[offset])
                    output_tokens = max(0, int(output.shape[-1]) - input_tokens)
                    records.append(
                        RequestMetric(
                            request_id=start_idx + offset,
                            ok=True,
                            latency_sec=per_request_latency,
                            ttft_sec=None,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        )
                    )
            except Exception as exc:
                elapsed = time.perf_counter() - start
                for offset, input_count in enumerate(input_counts):
                    records.append(
                        RequestMetric(
                            request_id=start_idx + offset,
                            ok=False,
                            latency_sec=elapsed,
                            ttft_sec=None,
                            input_tokens=int(input_count),
                            output_tokens=0,
                            error=str(exc),
                        )
                    )

    tag = args.tag or f"transformers_{args.quant}_in{args.input_len}_out{max_new_tokens}_bs{args.batch_size}"
    summary = summarize_request_metrics(
        records,
        {
            "backend": "transformers",
            "model": model_name,
            "quant": args.quant,
            "input_len_target": args.input_len,
            "output_len_target": max_new_tokens,
            "batch_size": args.batch_size,
            "peak_gpu_memory_mb": monitor.peak_used_mb,
            "mean_gpu_utilization": monitor.mean_gpu_utilization,
        },
    )
    write_jsonl(output_dir / f"{tag}.jsonl", records)
    write_json(output_dir / f"{tag}.summary.json", summary)
    print(f"Wrote {output_dir / f'{tag}.summary.json'}")


if __name__ == "__main__":
    main()

