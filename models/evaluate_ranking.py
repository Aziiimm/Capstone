"""
Offline ranking metrics for the item–item kNN recommender (same logic as recommender_cpu).

Trains *only* on a train split of interactions, then asks: for a held-out item the user
interacted with, does that item appear in the top-K recommendations built from the user’s
train history? Report Hit Rate@K (and optional MRR).

Uses sklearn + SciPy (CPU). Algorithm matches training in new_model.py / recommender_cpu;
cuML can differ slightly in tie-breaking but trends should align.

Example:
  cd models
  python evaluate_ranking.py --parquet ../output/tools_gpu_50k.parquet --top-k 10 --seed 42
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))


def _resolve_parquet_paths(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched: {pattern}")
    return paths


def _build_maps(df: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    users = df["reviewerID"].unique()
    items = df["asin"].unique()
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {a: i for i, a in enumerate(items)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    idx_to_item = {i: a for a, i in item_to_idx.items()}
    return user_to_idx, item_to_idx, idx_to_user, idx_to_item


def _df_to_train_csr(
    train_df: pd.DataFrame,
    user_to_idx: dict,
    item_to_idx: dict,
    n_users: int,
    n_items: int,
) -> sparse.csr_matrix:
    ui = train_df["reviewerID"].map(user_to_idx)
    ii = train_df["asin"].map(item_to_idx)
    ratings = train_df["rating"].astype(np.float32).values
    return sparse.coo_matrix(
        (ratings, (ii.values, ui.values)),
        shape=(n_items, n_users),
    ).tocsr()


def _recommend_item_indices(
    model: NearestNeighbors,
    mat: sparse.csr_matrix,
    history_item_idx: list[int],
    top_k: int,
) -> list[int]:
    """Mirror recommender_cpu.recommend index selection (ravel + slice)."""
    if not history_item_idx:
        return []
    q = np.asarray(history_item_idx, dtype=np.int64)
    _, indices = model.kneighbors(mat[q])
    flat = indices.ravel()
    out: list[int] = []
    for idx in flat[1 : top_k + 1]:
        out.append(int(idx))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hit Rate@K / MRR on held-out items (train-only fit; no leakage)."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Parquet file or glob (e.g. ../output/part.*.parquet)",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Per-user held-out fraction")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=5, help="kNN graph degree (fit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-eval-users",
        type=int,
        default=None,
        help="Cap users for a faster run (optional)",
    )
    parser.add_argument(
        "--min-user-items",
        type=int,
        default=3,
        help="Users need at least this many distinct items to be evaluated",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    paths = _resolve_parquet_paths(args.parquet)
    dfs = [pd.read_parquet(p, columns=["reviewerID", "asin", "rating"]) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["reviewerID", "asin"])

    user_to_idx, item_to_idx, _, _ = _build_maps(df)
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    # Per-user: distinct items (one interaction per user-item pair if duplicates, keep one rating — mean optional)
    user_items: dict[str, list[str]] = {}
    for uid, grp in df.groupby("reviewerID"):
        asins = grp["asin"].drop_duplicates().tolist()
        if len(asins) >= args.min_user_items:
            user_items[str(uid)] = asins

    users_list = list(user_items.keys())
    rng.shuffle(users_list)
    if args.max_eval_users:
        users_list = users_list[: args.max_eval_users]

    train_rows: list[dict] = []
    eval_cases: list[tuple[str, str, list[str]]] = []

    for uid in users_list:
        items = list(user_items[uid])
        rng.shuffle(items)
        n_hold = max(1, int(len(items) * args.test_fraction))
        test_asins = items[:n_hold]
        train_asins = items[n_hold:]
        if not train_asins:
            continue
        sub = df[(df["reviewerID"] == uid) & (df["asin"].isin(train_asins))].drop_duplicates(
            subset=["reviewerID", "asin"]
        )
        for _, r in sub.iterrows():
            train_rows.append(
                {"reviewerID": r["reviewerID"], "asin": r["asin"], "rating": float(r["rating"])}
            )
        for held in test_asins:
            sub_h = df[(df["reviewerID"] == uid) & (df["asin"] == held)]
            if sub_h.empty:
                continue
            eval_cases.append((uid, held, train_asins))

    train_df = pd.DataFrame(train_rows)
    if train_df.empty:
        raise SystemExit("No training rows after split; lower --min-user-items or check data.")

    mat = _df_to_train_csr(train_df, user_to_idx, item_to_idx, n_users, n_items)
    # k for kneighbors must be <= n_items and typically > 1
    k_fit = min(max(args.neighbors, 2), max(2, n_items))
    model = NearestNeighbors(n_neighbors=k_fit, metric="cosine")
    model.fit(mat)

    hits = 0
    rr_sum = 0.0
    n = 0
    for uid, held_asin, train_asins in eval_cases:
        hist_idx = [item_to_idx[a] for a in train_asins if a in item_to_idx]
        held_idx = item_to_idx.get(held_asin)
        if held_idx is None or not hist_idx:
            continue
        rec_idx = _recommend_item_indices(model, mat, hist_idx, args.top_k)
        n += 1
        if held_idx in rec_idx:
            hits += 1
            rank = rec_idx.index(held_idx) + 1
            rr_sum += 1.0 / rank
        else:
            rr_sum += 0.0

    if n == 0:
        raise SystemExit("No evaluation cases; increase data or loosen filters.")

    print(f"Cases evaluated: {n}")
    print(f"Hit Rate@{args.top_k}: {hits / n:.4f}")
    print(f"MRR@{args.top_k}: {rr_sum / n:.4f}")


if __name__ == "__main__":
    main()
