"""
Orchestrator: load -> filter -> to_parquet.
Updated to utilize LocalCUDACluster for distributed multi-GPU processing.
"""
from __future__ import annotations

import argparse
import json
import os
import time

# Distributed GPU imports
try:
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    HAS_DASK_CUDA = True
except ImportError:
    HAS_DASK_CUDA = False

from data.load import load
from data.filter import filter_counts
from data.to_parquet import to_parquet

def setup_cluster():
    """
    Initializes a multi-GPU cluster.
    Allocates a 7GB RMM pool per GPU to prevent OOM errors.
    """
    print("Initializing LocalCUDACluster...")
    cluster = LocalCUDACluster(
        rmm_pool_size="7GB",       # Allocates 7GB per GPU worker
        device_memory_limit="15GB" # Prevents spilling out of physical VRAM
    )
    client = Client(cluster)
    print(f"Cluster active! Dashboard link: {client.dashboard_link}")
    print(f"Workers (GPUs) connected: {len(client.scheduler_info()['workers'])}")
    return client

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data pipeline: load JSONL -> filter (user/item counts) -> Parquet."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--blocksize", default="64MB")
    parser.add_argument("--timings-file", default=None)
    args = parser.parse_args()

    client = None
    if args.gpu:
        if HAS_DASK_CUDA:
            client = setup_cluster()
        else:
            print("WARNING: dask_cuda not found. Running on single GPU/CPU without distributed cluster.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Update backend label for benchmarking
    backend = "gpu-cluster" if (args.gpu and HAS_DASK_CUDA) else ("gpu" if args.gpu else "cpu")
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
    timings["load_s"] = time.perf_counter() - t0
    print(f"  Load: {timings['load_s']:.2f}s")

    # Stage 2: Filter
    t0 = time.perf_counter()
    print("Filtering (user >= 6 reviews, item >= 11 reviews)...")
    ddf = filter_counts(ddf)
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

    # Clean up cluster connection to free up the GPUs
    if client:
        print("Shutting down cluster workers...")
        client.close()

if __name__ == "__main__":
    main()
