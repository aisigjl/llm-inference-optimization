#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--out", default="results/raw/gpu_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(args.gpu_index)
    try:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timestamp", "used_mb", "total_mb", "gpu_util", "mem_util"],
            )
            writer.writeheader()
            while True:
                mem = nvmlDeviceGetMemoryInfo(handle)
                util = nvmlDeviceGetUtilizationRates(handle)
                writer.writerow(
                    {
                        "timestamp": time.time(),
                        "used_mb": mem.used / 1024**2,
                        "total_mb": mem.total / 1024**2,
                        "gpu_util": util.gpu,
                        "mem_util": util.memory,
                    }
                )
                f.flush()
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"Wrote {out}")
    finally:
        nvmlShutdown()


if __name__ == "__main__":
    main()

