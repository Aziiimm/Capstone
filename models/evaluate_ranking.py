"""
Offline ranking metrics for the item–item kNN recommender (same logic as recommender_cpu).

Trains *only* on a train split of interactions, then asks: for a held-out item the user
interacted with, does that item appear in the top-K recommendations built from the user’s
train history? Report Hit Rate@K and MRR.

Also supports --split temporal (requires timestamp in Parquet) and popularity / random
baselines on the same evaluation cases.

Example:
  cd models
  python evaluate_ranking.py --parquet ../output/tools_gpu_50k.parquet --top-k 10 --seed 42
  python evaluate_ranking.py --parquet ../output/foo.parquet --split temporal --top-k 10
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

from neighbor_merge import merge_knn_scores


def _resolve_parquet_paths(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched: {pattern}")
    return paths


def _parquet_columns(paths: list[str]) -> list[str]:
    """Columns present in the first file (for optional timestamp)."""
    s = pd.read_parquet(paths[0])
    return list(s.columns)


def _parquet_names(path: str) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.read_schema(path).names)


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
    if not history_item_idx:
        return []
    q = np.asarray(history_item_idx, dtype=np.int64)
    distances, indices = model.kneighbors(mat[q])
    return merge_knn_scores(distances, indices, q, top_k)


def _asins_ordered_temporal(udf: pd.DataFrame) -> list[str]:
    """Unique ASINs ordered by last interaction time (ascending → oldest first)."""
    ts_col = "timestamp"
    if ts_col not in udf.columns:
        raise ValueError("timestamp column missing")
    g = udf[["asin", ts_col]].dropna(subset=[ts_col])
    if g.empty:
        return []
    last_ts = g.groupby("asin", as_index=False)[ts_col].max()
    last_ts = last_ts.sort_values(ts_col)
    return last_ts["asin"].astype(str).tolist()


def _accumulate_metrics(
    eval_cases: list[tuple[str, str, list[str]]],
    item_to_idx: dict,
    top_k: int,
    recommend_fn,
) -> tuple[int, float, int]:
    hits = 0
    rr_sum = 0.0
    n = 0
    for _, held_asin, train_asins in eval_cases:
        hist_idx = [item_to_idx[a] for a in train_asins if a in item_to_idx]
        held_idx = item_to_idx.get(held_asin)
        if held_idx is None or not hist_idx:
            continue
        rec_idx = recommend_fn(hist_idx, held_idx, train_asins)
        n += 1
        if held_idx in rec_idx:
            hits += 1
            rr_sum += 1.0 / (rec_idx.index(held_idx) + 1)
    return hits, rr_sum, n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hit Rate@K / MRR on held-out items (train-only fit; no leakage)."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Parquet file or glob (e.g. ../output/part.*.parquet)",
    )
    parser.add_argument(
        "--split",
        choices=("random", "temporal"),
        default="random",
        help="random: shuffle user items; temporal: hold out most-recent fraction (needs timestamp).",
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
    parser.add_argument(
        "--no-center-users",
        action="store_true",
        help="Disable per-user mean subtraction on train ratings (default: center, matches new_model.py).",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip popularity and random baselines (faster).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    paths = _resolve_parquet_paths(args.parquet)
    cols = _parquet_columns(paths)
    read_cols = ["reviewerID", "asin", "rating"]
    if args.split == "temporal":
        if "timestamp" not in cols:
            raise SystemExit(
                "Temporal split requires column 'timestamp' in Parquet. "
                "Use --split random or rebuild pipeline output with timestamp."
            )
        read_cols.append("timestamp")

    dfs = []
    for p in paths:
        names = _parquet_names(p)
        use = [c for c in read_cols if c in names]
        dfs.append(pd.read_parquet(p, columns=use))
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["reviewerID", "asin"])

    if args.split == "temporal" and "timestamp" not in df.columns:
        raise SystemExit("Temporal split: timestamp column not found after loading Parquet.")

    user_to_idx, item_to_idx, _, _ = _build_maps(df)
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    user_items: dict = {}
    for uid, grp in df.groupby("reviewerID"):
        asins = grp["asin"].drop_duplicates().tolist()
        if len(asins) >= args.min_user_items:
            user_items[uid] = asins

    users_list = list(user_items.keys())
    rng.shuffle(users_list)
    if args.max_eval_users:
        users_list = users_list[: args.max_eval_users]

    train_rows: list[dict] = []
    eval_cases: list[tuple[str, str, list[str]]] = []

    for uid in users_list:
        udf = df[df["reviewerID"] == uid]

        if args.split == "temporal":
            ordered = _asins_ordered_temporal(udf)
            if len(ordered) < args.min_user_items:
                continue
        else:
            items = list(user_items[uid])
            rng.shuffle(items)
            ordered = items

        n_hold = max(1, int(len(ordered) * args.test_fraction))
        if n_hold >= len(ordered):
            n_hold = max(1, len(ordered) - 1)

        if args.split == "temporal":
            train_asins = ordered[:-n_hold]
            test_asins = ordered[-n_hold:]
        else:
            test_asins = ordered[:n_hold]
            train_asins = ordered[n_hold:]

        if not train_asins:
            continue

        sub = udf[(udf["asin"].isin(train_asins))].drop_duplicates(subset=["reviewerID", "asin"])
        for _, r in sub.iterrows():
            train_rows.append(
                {"reviewerID": r["reviewerID"], "asin": r["asin"], "rating": float(r["rating"])}
            )
        for held in test_asins:
            sub_h = udf[udf["asin"] == held]
            if sub_h.empty:
                continue
            eval_cases.append((uid, held, train_asins))

    train_df = pd.DataFrame(train_rows)
    if train_df.empty:
        raise SystemExit("No training rows after split; lower --min-user-items or check data.")

    if not args.no_center_users:
        train_df = train_df.copy()
        train_df["rating"] = train_df.groupby("reviewerID")["rating"].transform(lambda x: x - x.mean())

    mat = _df_to_train_csr(train_df, user_to_idx, item_to_idx, n_users, n_items)
    k_fit = min(max(args.neighbors, 2), max(2, n_items))
    model = NearestNeighbors(n_neighbors=k_fit, metric="cosine")
    model.fit(mat)

    # --- Popularity ranking from train only (asin frequency) ---
    pop_rank_asins = train_df.groupby("asin").size().sort_values(ascending=False).index.tolist()

    def rec_knn(hist_idx: list[int], _held: int, _train_asins: list[str]) -> list[int]:
        return _recommend_item_indices(model, mat, hist_idx, args.top_k)

    def rec_popular(hist_idx: list[int], _held: int, train_asins: list[str]) -> list[int]:
        blocked = {item_to_idx[a] for a in train_asins if a in item_to_idx}
        out: list[int] = []
        for a in pop_rank_asins:
            ix = item_to_idx.get(a)
            if ix is None or ix in blocked:
                continue
            out.append(ix)
            if len(out) >= args.top_k:
                break
        return out

    def rec_random(hist_idx: list[int], _held: int, train_asins: list[str]) -> list[int]:
        blocked = set(hist_idx)
        pool = [i for i in range(n_items) if i not in blocked]
        rng.shuffle(pool)
        return pool[: args.top_k]

    print(f"Split: {args.split} | Scored neighbor merge: on | User-centered train: {not args.no_center_users}")

    h, rr, n = _accumulate_metrics(eval_cases, item_to_idx, args.top_k, rec_knn)
    if n == 0:
        raise SystemExit("No evaluation cases; increase data or loosen filters.")

    print(f"\n--- kNN (item–item) ---")
    print(f"Cases evaluated: {n}")
    print(f"Hit Rate@{args.top_k}: {h / n:.4f}")
    print(f"MRR@{args.top_k}: {rr / n:.4f}")

    if not args.skip_baselines:
        hp, rrp, _ = _accumulate_metrics(eval_cases, item_to_idx, args.top_k, rec_popular)
        print(f"\n--- Popularity baseline (train-set frequency, excluding user train items) ---")
        print(f"Hit Rate@{args.top_k}: {hp / n:.4f}")
        print(f"MRR@{args.top_k}: {rrp / n:.4f}")

        hr, rrr, _ = _accumulate_metrics(eval_cases, item_to_idx, args.top_k, rec_random)
        print(f"\n--- Random baseline (uniform among items, excluding user train items) ---")
        print(f"Hit Rate@{args.top_k}: {hr / n:.4f}")
        print(f"MRR@{args.top_k}: {rrr / n:.4f}")


if __name__ == "__main__":
    main()
