"""
Leave-one-out accuracy evaluation for the GPU recommender.

For each sampled user with >= MIN_HISTORY ratings:
  1. Hold out one of their rated items as the ground-truth target.
  2. Pass the remaining items as `history` into recommender.recommend().
  3. Ask the model for top-K recommendations.
  4. Score:
       Hit@K    -- 1 if target appears in top-K, else 0
       NDCG@K   -- 1 / log2(rank + 1) if target found at `rank`, else 0
       MRR      -- 1 / rank if target appears in top largest-K, averaged
       Latency  -- wall time of the .recommend() call

Run on the GPU box (same env that loads the pickle):

    cd ~/Capstone
    PYTHONPATH=$PYTHONPATH:$(pwd)/models \
        python models/evaluate.py \
        --model models/full_recommender.pkl \
        --users 1000 \
        --ks 5 10 20

The script holds the model on GPU but does the user-sampling and metric
accumulation on CPU (numpy), so VRAM stays roughly where uvicorn would put it.
"""
from __future__ import annotations

import argparse
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np

try:
    import cupy as cp
except Exception as exc:
    raise SystemExit(
        f"cupy not importable — run this on the GPU box where the model was "
        f"trained. ({exc})"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="models/full_recommender.pkl",
                   help="Path to the pickled AmazonRecommenderGPU.")
    p.add_argument("--users", type=int, default=1000,
                   help="Number of users to sample for evaluation.")
    p.add_argument("--min-history", type=int, default=5,
                   help="Skip users with fewer than this many ratings.")
    p.add_argument("--max-history", type=int, default=50,
                   help="Cap each user's history at this size to bound "
                        "inference cost. 0 disables the cap.")
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20],
                   help="K values to evaluate (e.g. --ks 5 10 20).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    print(f"Loading recommender from {model_path} ...")
    t0 = time.perf_counter()
    with open(model_path, "rb") as f:
        rec = pickle.load(f)
    print(f"  loaded in {time.perf_counter() - t0:.2f}s "
          f"({len(rec.title_map):,} items)")

    # The sparse matrix is shape (n_items, n_users). Transpose to CSC so each
    # column-slice gives one user's rated items in O(1) via indptr/indices.
    print("Building user → item map (CSR → CSC) ...")
    t0 = time.perf_counter()
    csc = rec.sparse_matrix.tocsc()
    indptr = cp.asnumpy(csc.indptr)
    indices = cp.asnumpy(csc.indices)
    n_items, n_users = csc.shape
    print(f"  done in {time.perf_counter() - t0:.2f}s "
          f"(n_users={n_users:,}, n_items={n_items:,})")

    rating_counts = np.diff(indptr)
    eligible = np.where(rating_counts >= args.min_history)[0]
    print(f"Eligible users (≥{args.min_history} ratings): {len(eligible):,}")
    if len(eligible) == 0:
        raise SystemExit("Lower --min-history; no users qualify.")

    n_sample = min(args.users, len(eligible))
    sampled = np.random.choice(eligible, size=n_sample, replace=False)
    print(f"Sampling {n_sample:,} users for evaluation.\n")

    largest_k = max(args.ks)
    metrics = {k: {"hits": 0, "ndcg": 0.0} for k in args.ks}
    rr_sum = 0.0
    latencies_ms: list[float] = []
    skipped_inference = 0
    skipped_unknown_asin = 0

    print(f"Evaluating with K ∈ {args.ks}, largest_k={largest_k} ...")
    eval_start = time.perf_counter()

    for i, u in enumerate(sampled, 1):
        items = indices[indptr[u]:indptr[u + 1]].tolist()
        if len(items) < 2:
            continue

        target = random.choice(items)
        history = [x for x in items if x != target]
        if args.max_history and len(history) > args.max_history:
            history = random.sample(history, args.max_history)

        try:
            t0 = time.perf_counter()
            recs = rec.recommend(history, top_k=largest_k)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
        except Exception:
            skipped_inference += 1
            continue

        rec_indices: list[int] = []
        for r in recs:
            idx = rec.asin_to_idx.get(r["asin"])
            if idx is None:
                skipped_unknown_asin += 1
                continue
            rec_indices.append(int(idx))

        rank = None
        for pos, ridx in enumerate(rec_indices, 1):
            if ridx == int(target):
                rank = pos
                break

        for k in args.ks:
            if rank is not None and rank <= k:
                metrics[k]["hits"] += 1
                metrics[k]["ndcg"] += 1.0 / math.log2(rank + 1)
        if rank is not None and rank <= largest_k:
            rr_sum += 1.0 / rank

        if i % 100 == 0:
            elapsed = time.perf_counter() - eval_start
            print(f"  {i}/{n_sample} users  ({elapsed:.1f}s elapsed)")

    eval_elapsed = time.perf_counter() - eval_start
    n_evaluated = len(latencies_ms)
    if n_evaluated == 0:
        raise SystemExit("No successful evaluations completed.")

    print()
    print("=" * 60)
    print(f"Evaluated: {n_evaluated:,} users in {eval_elapsed:.1f}s")
    print(f"Skipped — inference errors: {skipped_inference}, "
          f"unknown-asin recs: {skipped_unknown_asin}")
    print("-" * 60)
    print(f"{'K':>4}  {'Hit@K':>10}  {'NDCG@K':>10}")
    for k in args.ks:
        hit = metrics[k]["hits"] / n_evaluated
        ndcg = metrics[k]["ndcg"] / n_evaluated
        print(f"{k:>4}  {hit:>10.4f}  {ndcg:>10.4f}")
    print(f"\nMRR@{largest_k}: {rr_sum / n_evaluated:.4f}")

    lat = np.array(latencies_ms)
    print(f"\nLatency (ms) over {n_evaluated} queries:")
    print(f"  mean   = {lat.mean():>8.2f}")
    print(f"  median = {np.median(lat):>8.2f}")
    print(f"  p95    = {np.percentile(lat, 95):>8.2f}")
    print(f"  p99    = {np.percentile(lat, 99):>8.2f}")
    print(f"  min    = {lat.min():>8.2f}")
    print(f"  max    = {lat.max():>8.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
