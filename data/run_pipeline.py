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

try:
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    HAS_DASK_CUDA = True
except ImportError:
    HAS_DASK_CUDA = False


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
        default="128MB",
        help="Dask read block size (default 128MB)",
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

    client = None
    if args.gpu and HAS_DASK_CUDA:
        cluster = LocalCUDACluster(device_memory_limit="7GB")
        client = Client(cluster)
        import dask
        dask.config.set({"dataframe.shuffle.method": "tasks"})
        print("🚀 GPU Cluster started on RTX 3060 Ti")
        

    t0 = time.perf_counter()
    print("Loading...")
    ddf = load(
        args.source,
        meta=args.meta,
        limit=args.limit,
        use_gpu=args.gpu,
        blocksize=args.blocksize,
    )

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

    timings["total_s"] = timings["load_s"] + timings["filter_s"] + timings["to_parquet_s"]
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
    
    if client:
        client.close()


if __name__ == "__main__":
    main()
