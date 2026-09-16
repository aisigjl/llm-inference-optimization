#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-config", default="configs/benchmark.yaml")
    parser.add_argument("--mode", choices=["transformers", "api"], required=True)
    parser.add_argument("--quant", default="none")
    parser.add_argument("--backend", default="vllm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(ROOT / args.bench_config)
    matrix = cfg["matrix"]

    for input_len in matrix["input_lens"]:
        for output_len in matrix["output_lens"]:
            if args.mode == "transformers":
                for batch_size in matrix["batch_sizes"]:
                    cmd = [
                        sys.executable,
                        "scripts/run_transformers_baseline.py",
                        "--input-len",
                        str(input_len),
                        "--output-len",
                        str(output_len),
                        "--batch-size",
                        str(batch_size),
                        "--num-requests",
                        str(matrix["num_requests"]),
                        "--quant",
                        args.quant,
                    ]
                    print("Running:", " ".join(cmd), flush=True)
                    subprocess.run(cmd, cwd=ROOT, check=True)
            else:
                for concurrency in matrix["concurrencies"]:
                    cmd = [
                        sys.executable,
                        "scripts/run_api_benchmark.py",
                        "--input-len",
                        str(input_len),
                        "--output-len",
                        str(output_len),
                        "--concurrency",
                        str(concurrency),
                        "--num-requests",
                        str(matrix["num_requests"]),
                        "--backend",
                        args.backend,
                        "--quant",
                        args.quant,
                    ]
                    print("Running:", " ".join(cmd), flush=True)
                    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

