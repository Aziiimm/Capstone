"""
Train an implicit ALS recommender on Amazon Reviews parquet output.

One global model across ALL parquet shards (categories merged) so the factor
space captures cross-category signal. Same script runs on CPU or GPU via
--backend {cpu,gpu} -- identical algorithm, hyperparameters, and math.

Example:
    python -m models_als.train_als \\
        --path "output/dev_*.parquet" \\
        --factors 64 --iterations 20 --alpha 40 --regularization 0.05 \\
        --backend gpu \\
        --output models_als/als_recommender \\
        --timings-file output/timings_als_gpu.json
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from .recommender_als import AmazonRecommenderALS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train implicit ALS on Amazon Reviews parquet.")
    p.add_argument("--path", required=True,
                   help="Glob for parquet files, e.g. 'output/dev_*.parquet'.")
    p.add_argument("--factors", type=int, default=64)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--regularization", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=40.0,
                   help="Confidence weight: data = 1 + alpha * rating.")
    p.add_argument("--min-rating", type=float, default=4.0,
                   help="Drop interactions below this rating. Set to 0 to keep all.")
    p.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--output", default="models_als/als_recommender",
                   help="Directory to save the model and metadata.")
    p.add_argument("--timings-file", default=None,
                   help="Optional JSON file to write training timings.")
    return p.parse_args()


def load_interactions(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        print(f"  reading {p}")
        frames.append(pd.read_parquet(p, columns=["reviewerID", "asin", "rating", "product_title"]))
    df = pd.concat(frames, ignore_index=True)
    # rating arrives as object/str in some shards -- coerce to float and drop bad rows.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["rating"])
    if len(df) != before:
        print(f"  dropped {before - len(df):,} rows with non-numeric ratings")
    return df


def main() -> None:
    args = parse_args()
    timings: dict[str, float] = {}

    paths = sorted(glob.glob(args.path))
    if not paths:
        raise SystemExit(f"No parquet files matched: {args.path}")
    print(f"Found {len(paths)} parquet shard(s).")

    t0 = time.perf_counter()
    df = load_interactions(paths)
    timings["load_s"] = time.perf_counter() - t0
    print(f"Loaded {len(df):,} interactions in {timings['load_s']:.2f}s")

    if args.min_rating > 0:
        before = len(df)
        df = df[df["rating"] >= args.min_rating]
        print(f"After rating >= {args.min_rating}: {len(df):,} (dropped {before - len(df):,})")

    t0 = time.perf_counter()
    title_pairs = df[["asin", "product_title"]].drop_duplicates(subset=["asin"])
    user_codes, user_index = pd.factorize(df["reviewerID"], sort=False)
    item_codes, item_index = pd.factorize(df["asin"], sort=False)
    n_users, n_items = len(user_index), len(item_index)
    print(f"n_users={n_users:,}  n_items={n_items:,}  nnz={len(df):,}")

    confidence = (1.0 + args.alpha * df["rating"].to_numpy(dtype=np.float32)).astype(np.float32)
    user_item = sp.csr_matrix(
        (confidence, (user_codes.astype(np.int32), item_codes.astype(np.int32))),
        shape=(n_users, n_items),
    )

    asin_to_idx = {asin: i for i, asin in enumerate(item_index)}
    title_map = {
        asin_to_idx[a]: t
        for a, t in zip(title_pairs["asin"], title_pairs["product_title"])
        if a in asin_to_idx
    }
    timings["matrix_build_s"] = time.perf_counter() - t0
    print(f"Built user x item CSR in {timings['matrix_build_s']:.2f}s")

    use_gpu = args.backend == "gpu"
    print(f"Training ALS on {args.backend.upper()} "
          f"(factors={args.factors}, iters={args.iterations}, "
          f"reg={args.regularization}, alpha={args.alpha})")
    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        use_gpu=use_gpu,
    )

    t0 = time.perf_counter()
    model.fit(user_item, show_progress=True)
    timings["train_s"] = time.perf_counter() - t0
    print(f"Trained in {timings['train_s']:.2f}s")

    t0 = time.perf_counter()
    rec = AmazonRecommenderALS(
        model=model, user_item=user_item, title_map=title_map, asin_to_idx=asin_to_idx
    )
    rec.save(args.output)
    timings["save_s"] = time.perf_counter() - t0
    print(f"Saved recommender to {args.output} in {timings['save_s']:.2f}s")

    timings["total_s"] = sum(timings.values())
    print(f"\nTotal: {timings['total_s']:.2f}s  (backend={args.backend})")

    if args.timings_file:
        Path(args.timings_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.timings_file, "w") as f:
            json.dump({
                "backend": args.backend,
                "factors": args.factors,
                "iterations": args.iterations,
                "regularization": args.regularization,
                "alpha": args.alpha,
                "min_rating": args.min_rating,
                "n_users": int(n_users),
                "n_items": int(n_items),
                "nnz": int(user_item.nnz),
                "timings_s": timings,
            }, f, indent=2)
        print(f"Timings written to {args.timings_file}")


if __name__ == "__main__":
    main()
