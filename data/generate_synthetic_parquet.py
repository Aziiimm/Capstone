"""
Create a small reproducible Parquet for dev/benchmark (no Amazon download required).
Columns match data/to_parquet.FINAL_COLUMNS.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

FINAL_COLUMNS = [
    "reviewerID",
    "asin",
    "rating",
    "reviewText",
    "timestamp",
    "product_title",
    "main_category",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Write synthetic reviews Parquet.")
    ap.add_argument(
        "--output",
        default="output/dev_synthetic.parquet",
        help="Output Parquet path",
    )
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--items", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    users = [f"u{i}" for i in range(args.users)]
    items = [f"B{i:05d}" for i in range(args.items)]

    rows: list[dict] = []
    ts = 0
    for u in users:
        nRated = int(rng.integers(8, 16))
        picks = rng.choice(args.items, size=nRated, replace=False)
        for j in picks.astype(int):
            ts += 1
            asin = items[j]
            rows.append(
                {
                    "reviewerID": u,
                    "asin": asin,
                    "rating": float(rng.integers(3, 6)),
                    "reviewText": "synthetic review",
                    "timestamp": ts,
                    "product_title": f"Product {asin}",
                    "main_category": "Synthetic",
                }
            )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = pd.DataFrame(rows)[FINAL_COLUMNS]
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
