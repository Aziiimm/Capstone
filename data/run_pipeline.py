"""
Orchestrator: load -> filter -> to_parquet.
Updated to allocate 7GB of VRAM for the RMM pool.
"""
from __future__ import annotations

import argparse
import json
import os
import time

# RAPIDS imports for memory management
try:
    import rmm
    HAS_RMM = True
except ImportError:
    HAS_RMM = False

from data.load import load
from data.filter import filter_counts
from data.to_parquet import to_parquet

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data pipeline: load JSONL -> filter (user/item counts) -> Parquet."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to local review JSONL/JSONL.GZ (e.g. dataset/Tools_and_Home_Improvement.jsonl.gz)",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="Path to item metadata JSONL.GZ (e.g. dataset/meta_Tools_and_Home_Improvement.jsonl.gz). Optional; if set, join and add product_title, main_category.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output Parquet path (file or directory)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: limit rows (for testing). Omit for full dataset.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU (Dask-cuDF) if available",
    )
    parser.add_argument(
        "--blocksize",
        default="64MB",
        help="Dask read block size (default 64MB)",
    )
    parser.add_argument(
        "--timings-file",
        default=None,
        help="Optional: write stage timings (seconds) to this JSON file for comparison across runs.",
    )
    parser.add_argument(
        "--min-user-reviews",
        type=int,
        default=None,
        help="Override min reviews per user for filter_counts (default: 6).",
    )
    parser.add_argument(
        "--min-item-reviews",
        type=int,
        default=None,
        help="Override min reviews per item for filter_counts (default: 11).",
    )
    args = parser.parse_args()

    if args.gpu and HAS_RMM:
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=int(7e9),
            managed_memory=True,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    backend = "gpu" if args.gpu else "cpu"
    timings: dict[str, float] = {}

    # Stage 1: Load
    t0 = time.perf_counter()
    print(f"Loading (Backend: {backend})...")
    ddf = load(
        args.source,
        meta=args.meta,
        limit=args.limit,
        use_gpu=args.gpu,
        blocksize=args.blocksize,
    )

    # Stage 1: 
    timings["load_s"] = time.perf_counter() - t0
    print(f"  Load: {timings['load_s']:.2f}s")

    # Stage 2: Filter
    t0 = time.perf_counter()
    print("Filtering (user >= 6 reviews, item >= 11 reviews)...")
    ddf = filter_counts(
        ddf,
        min_reviews_per_user=args.min_user_reviews,
        min_reviews_per_item=args.min_item_reviews,
    )
    timings["filter_s"] = time.perf_counter() - t0
    print(f"  Filter: {timings['filter_s']:.2f}s")

    # Stage 3: Write
    t0 = time.perf_counter()
    print("Writing Parquet...")
    to_parquet(ddf, args.output)
    timings["to_parquet_s"] = time.perf_counter() - t0
    print(f"  To Parquet: {timings['to_parquet_s']:.2f}s")

    timings["total_s"] = sum(v for k, v in timings.items() if "_s" in k)
    print(f"Done: {args.output} (total {timings['total_s']:.2f}s, backend={backend})")

    if args.timings_file:
        with open(args.timings_file, "w") as f:
            json.dump({"backend": backend, "timings_s": timings}, f, indent=2)

if __name__ == "__main__":
    main()