"""
Time-based holdout evaluation for the implicit ALS recommender.

For each user with >= --min-history interactions:
  1. Sort by timestamp.
  2. Hold out the most recent interaction.
  3. Train ALS on everything else.
  4. Ask the model for top-K recs given the truncated history.
  5. Score Hit@K, NDCG@K, MRR.

This is the honest comparison the legacy evaluator skipped: held-out items were
never seen during training, so metrics are not contaminated.

Example:
    python -m models_als.evaluate_als \\
        --path "output/dev_*.parquet" \\
        --backend gpu \\
        --users 1000 --ks 5 10 20 \\
        --metrics-file output/metrics_als_gpu.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--path", required=True)
    p.add_argument("--factors", type=int, default=64)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--regularization", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=40.0)
    p.add_argument("--min-rating", type=float, default=4.0)
    p.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--users", type=int, default=1000,
                   help="Number of holdout users to evaluate.")
    p.add_argument("--min-history", type=int, default=5)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metrics-file", default=None,
                   help="Optional JSON file to write metrics.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    paths = sorted(glob.glob(args.path))
    if not paths:
        raise SystemExit(f"No parquet files matched: {args.path}")

    t0 = time.perf_counter()
    frames = []
    for p in paths:
        print(f"  reading {p}")
        frames.append(pd.read_parquet(p, columns=["reviewerID", "asin", "rating", "timestamp"]))
    df = pd.concat(frames, ignore_index=True)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["rating", "timestamp"])
    if len(df) != before:
        print(f"  dropped {before - len(df):,} rows with bad rating/timestamp")
    print(f"Loaded {len(df):,} interactions in {time.perf_counter() - t0:.2f}s")

    if args.min_rating > 0:
        df = df[df["rating"] >= args.min_rating]
        print(f"After rating >= {args.min_rating}: {len(df):,}")

    user_codes, user_index = pd.factorize(df["reviewerID"], sort=False)
    item_codes, item_index = pd.factorize(df["asin"], sort=False)
    df = df.assign(
        user_idx=user_codes.astype(np.int32),
        item_idx=item_codes.astype(np.int32),
    )
    n_users, n_items = len(user_index), len(item_index)
    print(f"n_users={n_users:,}  n_items={n_items:,}")

    t0 = time.perf_counter()
    df = df.sort_values(["user_idx", "timestamp"], kind="stable")
    last_idx = df.groupby("user_idx").tail(1).index
    user_counts = df.groupby("user_idx").size()
    eligible_users = set(user_counts[user_counts >= args.min_history].index.tolist())

    holdout_mask = df.index.isin(last_idx)
    df_train = df[~holdout_mask]
    df_holdout = df[holdout_mask]
    df_holdout = df_holdout[df_holdout["user_idx"].isin(eligible_users)]
    print(f"Train rows: {len(df_train):,}  Holdout users: {len(df_holdout):,}  "
          f"({time.perf_counter() - t0:.2f}s to split)")

    confidence = (1.0 + args.alpha * df_train["rating"].to_numpy(dtype=np.float32)).astype(np.float32)
    user_item = sp.csr_matrix(
        (confidence,
         (df_train["user_idx"].to_numpy(dtype=np.int32),
          df_train["item_idx"].to_numpy(dtype=np.int32))),
        shape=(n_users, n_items),
    )

    use_gpu = args.backend == "gpu"
    print(f"Training ALS on {args.backend.upper()} "
          f"(factors={args.factors}, iters={args.iterations})")
    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        use_gpu=use_gpu,
    )
    t0 = time.perf_counter()
    model.fit(user_item, show_progress=True)
    train_s = time.perf_counter() - t0
    print(f"Trained in {train_s:.2f}s")

    holdout_arr = df_holdout[["user_idx", "item_idx"]].to_numpy()
    n_sample = min(args.users, len(holdout_arr))
    sample_idx = np.random.choice(len(holdout_arr), size=n_sample, replace=False)
    sampled = holdout_arr[sample_idx]
    print(f"Evaluating on {n_sample:,} sampled holdout users.")

    largest_k = max(args.ks)
    metrics = {k: {"hits": 0, "ndcg": 0.0} for k in args.ks}
    rr_sum = 0.0
    latencies_ms: list[float] = []

    eval_start = time.perf_counter()
    for i, (u, target) in enumerate(sampled, 1):
        u = int(u)
        target = int(target)
        t0 = time.perf_counter()
        ids, _ = model.recommend(
            userid=u,
            user_items=user_item[u],
            N=largest_k,
            filter_already_liked_items=True,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        rec_ids = ids.tolist()
        rank = None
        for pos, ridx in enumerate(rec_ids, 1):
            if int(ridx) == target:
                rank = pos
                break

        for k in args.ks:
            if rank is not None and rank <= k:
                metrics[k]["hits"] += 1
                metrics[k]["ndcg"] += 1.0 / math.log2(rank + 1)
        if rank is not None and rank <= largest_k:
            rr_sum += 1.0 / rank

        if i % 100 == 0:
            print(f"  {i}/{n_sample} users  ({time.perf_counter() - eval_start:.1f}s)")

    eval_elapsed = time.perf_counter() - eval_start
    n_eval = len(latencies_ms)

    print()
    print("=" * 60)
    print(f"Evaluated: {n_eval:,} users in {eval_elapsed:.1f}s")
    print(f"Backend: {args.backend.upper()}  |  Train time: {train_s:.2f}s")
    print("-" * 60)
    print(f"{'K':>4}  {'Hit@K':>10}  {'NDCG@K':>10}")
    results = {}
    for k in args.ks:
        hit = metrics[k]["hits"] / n_eval
        ndcg = metrics[k]["ndcg"] / n_eval
        results[f"hit_at_{k}"] = hit
        results[f"ndcg_at_{k}"] = ndcg
        print(f"{k:>4}  {hit:>10.4f}  {ndcg:>10.4f}")
    mrr = rr_sum / n_eval
    print(f"\nMRR@{largest_k}: {mrr:.4f}")
    results[f"mrr_at_{largest_k}"] = mrr

    lat = np.array(latencies_ms)
    print(f"\nLatency (ms) over {n_eval} queries:")
    print(f"  mean={lat.mean():.2f}  median={np.median(lat):.2f}  "
          f"p95={np.percentile(lat, 95):.2f}  p99={np.percentile(lat, 99):.2f}")
    print("=" * 60)

    if args.metrics_file:
        Path(args.metrics_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_file, "w") as f:
            json.dump({
                "backend": args.backend,
                "factors": args.factors,
                "iterations": args.iterations,
                "regularization": args.regularization,
                "alpha": args.alpha,
                "min_rating": args.min_rating,
                "n_users": int(n_users),
                "n_items": int(n_items),
                "nnz_train": int(user_item.nnz),
                "n_evaluated": int(n_eval),
                "train_s": train_s,
                "eval_s": eval_elapsed,
                "metrics": results,
                "latency_ms": {
                    "mean": float(lat.mean()),
                    "median": float(np.median(lat)),
                    "p95": float(np.percentile(lat, 95)),
                    "p99": float(np.percentile(lat, 99)),
                },
            }, f, indent=2)
        print(f"Metrics written to {args.metrics_file}")


if __name__ == "__main__":
    main()
