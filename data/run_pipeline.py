"""
Orchestrator: load -> filter -> to_parquet (data_clean entry point).
Tracks wall-clock time per stage for performance comparison (CPU vs GPU, runs).
"""
from __future__ import annotations

import argparse
import json
import os
import time

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
        "--workers",
        type=int,
        default=0,
        help=(
            "Optional: number of Dask worker processes to use for the CPU path. "
            "0 (default) keeps the single-process threaded scheduler."
        ),
    )
    parser.add_argument(
        "--timings-file",
        default=None,
        help="Optional: write stage timings (seconds) to this JSON file for comparison across runs.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    backend = "gpu" if args.gpu else "cpu"
    timings: dict[str, float] = {}

    cluster = None
    client = None
    try:
        if not args.gpu and args.workers and args.workers > 0:
            try:
                from dask.distributed import Client, LocalCluster
            except ImportError as exc:
                raise SystemExit(
                    "Requested --workers for multi-process Dask execution, but "
                    "dask.distributed is not available. Install it via "
                    "`pip install 'dask[distributed]'` or `pip install 'dask[complete]'`."
                ) from exc
            cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
            client = Client(cluster)
            print(
                f"Using local Dask distributed scheduler with {args.workers} workers "
                "(threads_per_worker=1) for CPU pipeline."
            )
        elif args.gpu and args.workers and args.workers > 0:
            print("Warning: --workers is ignored when --gpu is set; using GPU scheduler as-is.")

        t0 = time.perf_counter()
        print("Loading...")
        ddf = load(
            args.source,
            meta=args.meta,
            limit=args.limit,
            use_gpu=args.gpu,
            blocksize=args.blocksize,
        )
        timings["load_s"] = time.perf_counter() - t0
        print(f"  Load: {timings['load_s']:.2f}s")

        t0 = time.perf_counter()
        print("Filtering (user >= 6 reviews, item >= 11 reviews)...")
        ddf = filter_counts(ddf)
        timings["filter_s"] = time.perf_counter() - t0
        print(f"  Filter: {timings['filter_s']:.2f}s")

        t0 = time.perf_counter()
        print("Writing Parquet...")
        to_parquet(ddf, args.output)
        timings["to_parquet_s"] = time.perf_counter() - t0
        print(f"  To Parquet: {timings['to_parquet_s']:.2f}s")

        timings["total_s"] = (
            timings["load_s"] + timings["filter_s"] + timings["to_parquet_s"]
        )
        print(f"Done: {args.output} (total {timings['total_s']:.2f}s, backend={backend})")

        if args.timings_file:
            out = {
                "backend": backend,
                "source": args.source,
                "meta": args.meta,
                "output": args.output,
                "limit": args.limit,
                "timings_s": timings,
            }
            with open(args.timings_file, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Timings written to {args.timings_file}")
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
