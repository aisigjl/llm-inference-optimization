#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="results/tables/summary.csv")
    parser.add_argument("--fig-dir", default="results/figures")
    return parser.parse_args()


def label(row: pd.Series) -> str:
    return f"{row.get('backend', 'unknown')}-{row.get('quant', 'none')}"


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out: Path) -> None:
    if metric not in df.columns or df[metric].dropna().empty:
        return
    data = df.copy()
    data["series"] = data.apply(label, axis=1)
    data["load"] = data.get("concurrency", data.get("batch_size", 1)).fillna(data.get("batch_size", 1))
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=data,
        x="load",
        y=metric,
        hue="series",
        style="input_len_target",
        marker="o",
    )
    plt.xlabel("Concurrency or batch size")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_metric(df, "output_tokens_per_sec", "Output tokens/s", fig_dir / "output_tokens_per_sec.png")
    plot_metric(df, "latency_p95_sec", "P95 latency (s)", fig_dir / "latency_p95_sec.png")
    plot_metric(df, "ttft_p95_sec", "P95 TTFT (s)", fig_dir / "ttft_p95_sec.png")
    plot_metric(df, "peak_gpu_memory_mb", "Peak GPU memory (MB)", fig_dir / "peak_gpu_memory_mb.png")
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()

