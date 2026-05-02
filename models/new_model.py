"""Train item–item kNN recommender on Parquet (CPU sklearn or GPU cuML)."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from build_matrix import build_matrix_from_parquet
from recommender_cpu import AmazonRecommenderCPU
from sklearn.neighbors import NearestNeighbors as SKNearestNeighbors


def _train_cpu(artifacts, n_neighbors: int) -> AmazonRecommenderCPU:
    model = SKNearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
    )
    model.fit(artifacts.sparse_csr)
    return AmazonRecommenderCPU(model, artifacts.sparse_csr, artifacts.title_map)


def _train_gpu(artifacts, n_neighbors: int, rmm_pool_gb: float):
    import rmm
    import cupyx.scipy.sparse as cpsparse
    from cuml.neighbors import NearestNeighbors as CUNearestNeighbors

    from recommender import AmazonRecommenderGPU

    pool_bytes = max(int(rmm_pool_gb * 1e9), 256_000_000)
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=pool_bytes,
        managed_memory=True,
    )
    sparse_gpu = cpsparse.csr_matrix(artifacts.sparse_csr)
    model = CUNearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    model.fit(sparse_gpu)
    return AmazonRecommenderGPU(model, sparse_gpu, artifacts.title_map)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Amazon item–item kNN recommender (CPU or GPU)."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Glob for Parquet files (e.g. 'output/dev_*.parquet')",
    )
    parser.add_argument("--neighbors", type=int, default=10, help="k for kNN")
    parser.add_argument(
        "--output",
        type=str,
        default="models/full_recommender.pkl",
        help="Pickle output path",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="cpu = sklearn; gpu = cuML (requires RAPIDS)",
    )
    parser.add_argument(
        "--rmm-pool-gb",
        type=float,
        default=2.0,
        help="RMM initial pool size in GB (GPU only). Lower for 6GB cards.",
    )
    parser.add_argument(
        "--min-user-reviews",
        type=int,
        default=1,
        help="Drop users with fewer reviews after load (1 = no extra filter).",
    )
    parser.add_argument(
        "--min-item-reviews",
        type=int,
        default=1,
        help="Drop items with fewer reviews after load (1 = no extra filter).",
    )
    parser.add_argument(
        "--rating-mode",
        choices=("raw", "binary"),
        default="raw",
        help="binary = implicit 1/0 from rating sign (often improves sparse signals).",
    )
    args = parser.parse_args()

    print(f"Loading matrix from glob: {args.path}")
    artifacts = build_matrix_from_parquet(
        args.path,
        min_reviews_per_user=args.min_user_reviews,
        min_reviews_per_item=args.min_item_reviews,
        rating_mode=args.rating_mode,
    )
    print(
        f"  users={artifacts.n_users:,} items={artifacts.n_items:,} nnz={artifacts.nnz:,}"
    )

    if args.device == "cpu":
        print(f"Training sklearn NearestNeighbors (k={args.neighbors})...")
        recommender = _train_cpu(artifacts, args.neighbors)
    else:
        print(f"Training cuML NearestNeighbors (k={args.neighbors})...")
        recommender = _train_gpu(artifacts, args.neighbors, args.rmm_pool_gb)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "wb") as f:
        pickle.dump(recommender, f)

    print(f"SUCCESS: saved recommender ({args.device}) -> {args.output}")


if __name__ == "__main__":
    main()
