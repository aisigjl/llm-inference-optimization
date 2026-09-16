from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RequestMetric:
    request_id: int
    ok: bool
    latency_sec: float
    ttft_sec: float | None
    input_tokens: int
    output_tokens: int
    error: str | None = None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def summarize_request_metrics(
    records: list[RequestMetric],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ok_records = [r for r in records if r.ok]
    latencies = [r.latency_sec for r in ok_records]
    ttfts = [r.ttft_sec for r in ok_records if r.ttft_sec is not None]
    input_tokens = sum(r.input_tokens for r in ok_records)
    output_tokens = sum(r.output_tokens for r in ok_records)
    total_time = max(latencies) if latencies else 0.0

    summary: dict[str, Any] = {
        "num_requests": len(records),
        "num_success": len(ok_records),
        "num_error": len(records) - len(ok_records),
        "latency_mean_sec": statistics.mean(latencies) if latencies else None,
        "latency_p50_sec": percentile(latencies, 50),
        "latency_p90_sec": percentile(latencies, 90),
        "latency_p95_sec": percentile(latencies, 95),
        "latency_p99_sec": percentile(latencies, 99),
        "ttft_mean_sec": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50_sec": percentile(ttfts, 50),
        "ttft_p95_sec": percentile(ttfts, 95),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "output_tokens_per_sec": output_tokens / total_time if total_time > 0 else None,
        "total_tokens_per_sec": (input_tokens + output_tokens) / total_time if total_time > 0 else None,
    }
    if extra:
        summary.update(extra)
    return summary


def write_jsonl(path: str | Path, records: list[RequestMetric]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

