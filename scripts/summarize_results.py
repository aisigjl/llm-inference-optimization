#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--out", default="results/tables/summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    rows = []
    for path in sorted(raw_dir.glob("*.summary.json")):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row["source_file"] = path.name
        rows.append(row)
    if not rows:
        raise SystemExit(f"No summary files found under {raw_dir}")
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

