"""
Fair timing: same Parquet → same CSR → sklearn vs cuML fit + inference latency.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

import numpy as np

from build_matrix import build_matrix_from_parquet
from new_model import _train_cpu, _train_gpu


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark CPU vs GPU kNN recommender.")
    p.add_argument("--path", required=True, help="Parquet glob")
    p.add_argument("--neighbors", type=int, default=10)
    p.add_argument("--queries", type=int, default=32, help="Random item queries for inference")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--rmm-pool-gb", type=float, default=2.0)
    p.add_argument(
        "--rating-mode",
        choices=("raw", "binary"),
        default="raw",
    )
    args = p.parse_args()

    print("Building matrix (CPU)...")
    artifacts = build_matrix_from_parquet(args.path, rating_mode=args.rating_mode)
    rng = np.random.default_rng(42)
    n_items = artifacts.n_items
    query_items = rng.choice(n_items, size=min(args.queries, n_items), replace=False).tolist()

    # --- CPU ---
    t0 = time.perf_counter()
    cpu_rec = _train_cpu(artifacts, args.neighbors)
    cpu_fit_s = time.perf_counter() - t0

    inf_times_ms: list[float] = []
    for qi in query_items:
        t1 = time.perf_counter()
        _ = cpu_rec.recommend([int(qi)], top_k=args.top_k)
        inf_times_ms.append((time.perf_counter() - t1) * 1000.0)

    print("\n=== CPU (sklearn) ===")
    print(f"  fit_time_s: {cpu_fit_s:.4f}")
    print(f"  inference_ms_mean: {statistics.mean(inf_times_ms):.3f}")
    print(f"  inference_ms_stdev: {statistics.pstdev(inf_times_ms):.4f}")

    # --- GPU ---
    try:
        t0 = time.perf_counter()
        gpu_rec = _train_gpu(artifacts, args.neighbors, args.rmm_pool_gb)
        gpu_fit_s = time.perf_counter() - t0

        # warmup
        if query_items:
            _ = gpu_rec.recommend([int(query_items[0])], top_k=args.top_k)

        gpu_inf_ms: list[float] = []
        for qi in query_items:
            t1 = time.perf_counter()
            _ = gpu_rec.recommend([int(qi)], top_k=args.top_k)
            gpu_inf_ms.append((time.perf_counter() - t1) * 1000.0)

        print("\n=== GPU (cuML) ===")
        print(f"  fit_time_s: {gpu_fit_s:.4f}")
        print(f"  inference_ms_mean: {statistics.mean(gpu_inf_ms):.3f}")
        print(f"  inference_ms_stdev: {statistics.pstdev(gpu_inf_ms):.4f}")
    except Exception as e:
        print("\n=== GPU (cuML) ===")
        print(f"  skipped: {type(e).__name__}: {e}")

    print("\n=== Summary ===")
    print(f"  dataset: users={artifacts.n_users} items={artifacts.n_items} nnz={artifacts.nnz}")
    print(f"  k={args.neighbors} queries={len(query_items)} top_k={args.top_k}")


if __name__ == "__main__":
    main()
