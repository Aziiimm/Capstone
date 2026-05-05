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
  # Path may be a single .parquet file OR a directory of part-*.parquet from Dask/RAPIDS.
  python evaluate_ranking.py --parquet "../output/foo/part.*.parquet" --split temporal --top-k 10
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from neighbor_merge import merge_knn_scores


def _expand_to_parquet_files(paths: list[str]) -> list[str]:
    """Single Parquet file or a directory of part-*.parquet (Dask/cuDF output)."""
    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            parts = sorted(glob.glob(os.path.join(p, "*.parquet")))
            if not parts:
                parts = sorted(glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True))
            if not parts:
                raise SystemExit(f"No .parquet files under directory: {p}")
            out.extend(parts)
        else:
            out.append(p)
    return out


def _resolve_parquet_inputs(pattern: str) -> list[str]:
    matches = sorted(glob.glob(pattern))
    if not matches and os.path.lexists(pattern):
        matches = [os.path.normpath(pattern)]
    if not matches:
        raise SystemExit(f"No path matched: {pattern!r}")
    return _expand_to_parquet_files(matches)


def _parquet_columns_from_file(path: str) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


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
    *,
    implicit: bool = False,
) -> sparse.csr_matrix:
    ui = train_df["reviewerID"].map(user_to_idx)
    ii = train_df["asin"].map(item_to_idx)
    if implicit:
        ratings = np.ones(len(train_df), dtype=np.float32)
    else:
        ratings = train_df["rating"].astype(np.float32).values
    return sparse.coo_matrix(
        (ratings, (ii.values, ui.values)),
        shape=(n_items, n_users),
    ).tocsr()


def _bm25_weight_csr(mat: sparse.csr_matrix, k1: float = 100.0, b: float = 0.8) -> sparse.csr_matrix:
    """BM25 reweighting (rows=items, cols=users) — same idea as implicit.bm25_weight.

    Active users (high column nnz) get less idf weight; long item rows get
    saturated by the (k1, b) length normalization. Cosine kNN over this matrix
    typically beats raw counts on sparse implicit feedback.
    """
    if mat.nnz == 0:
        return mat.copy()
    m = mat.tocsr().astype(np.float64)
    n_rows, _ = m.shape
    col_nnz = np.asarray((m != 0).sum(axis=0)).flatten().astype(np.float64)
    idf = np.log((n_rows - col_nnz + 0.5) / (col_nnz + 0.5) + 1.0)
    row_sums = np.asarray(m.sum(axis=1)).flatten()
    avg_row = float(row_sums.mean()) if row_sums.size else 1.0
    if avg_row <= 0.0:
        avg_row = 1.0
    coo = m.tocoo()
    rows, cols, vals = coo.row, coo.col, coo.data
    norm = row_sums[rows] / avg_row
    new_vals = idf[cols] * ((k1 + 1.0) * vals) / (vals + k1 * (1.0 - b + b * norm))
    return sparse.coo_matrix((new_vals, (rows, cols)), shape=m.shape).tocsr().astype(np.float32)


def _recommend_item_indices(
    model: NearestNeighbors,
    mat: sparse.csr_matrix,
    history_item_idx: list[int],
    top_k: int,
    row_weights: np.ndarray | None = None,
) -> list[int]:
    if not history_item_idx:
        return []
    q = np.asarray(history_item_idx, dtype=np.int64)
    distances, indices = model.kneighbors(mat[q])
    return merge_knn_scores(distances, indices, q, top_k, row_weights=row_weights)


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
    for uid, held_asin, train_asins in eval_cases:
        hist_idx = [item_to_idx[a] for a in train_asins if a in item_to_idx]
        held_idx = item_to_idx.get(held_asin)
        if held_idx is None or not hist_idx:
            continue
        rec_idx = recommend_fn(uid, hist_idx, held_idx, train_asins)
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
    parser.add_argument(
        "--implicit-matrix",
        action="store_true",
        help="Train kNN on binary interactions (matrix values 1); skips user-centering.",
    )
    parser.add_argument(
        "--rating-weighted-merge",
        action="store_true",
        help="Weight each history row in scored merge by star rating/5 (often helps; try on your data).",
    )
    parser.add_argument(
        "--bm25-weighting",
        action="store_true",
        help="Reweight item-user matrix with BM25 before fitting kNN. Implies --implicit-matrix.",
    )
    parser.add_argument("--bm25-k1", type=float, default=100.0, help="BM25 saturation (default 100).")
    parser.add_argument("--bm25-b", type=float, default=0.8, help="BM25 length normalization (default 0.8).")
    parser.add_argument(
        "--model",
        choices=("knn", "svd"),
        default="knn",
        help="knn: item–item cosine kNN (default). svd: TruncatedSVD latent factors (item × user).",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=64,
        help="Latent dimension for --model svd (default 64; will be clipped to matrix size).",
    )
    args = parser.parse_args()

    if args.bm25_weighting:
        # BM25 expects nonnegative values; force implicit + skip centering.
        args.implicit_matrix = True
        args.no_center_users = True

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    paths = _resolve_parquet_inputs(args.parquet)
    cols = _parquet_columns_from_file(paths[0])
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

    rating_lookup: dict[tuple, float] = {}
    for _, r in train_df.iterrows():
        rating_lookup[(r["reviewerID"], r["asin"])] = float(r["rating"])

    train_fit = train_df.copy()
    if args.implicit_matrix:
        train_fit["rating"] = 1.0
    elif not args.no_center_users:
        train_fit["rating"] = train_fit.groupby("reviewerID")["rating"].transform(lambda x: x - x.mean())

    mat = _df_to_train_csr(
        train_fit,
        user_to_idx,
        item_to_idx,
        n_users,
        n_items,
        implicit=args.implicit_matrix,
    )
    if args.bm25_weighting:
        mat = _bm25_weight_csr(mat, k1=args.bm25_k1, b=args.bm25_b)

    if args.model == "knn":
        k_fit = min(max(args.neighbors, 2), max(2, n_items))
        model = NearestNeighbors(n_neighbors=k_fit, metric="cosine")
        model.fit(mat)
        item_factors = None
    else:
        max_components = max(1, min(mat.shape) - 1)
        comps = max(1, min(args.svd_components, max_components))
        svd = TruncatedSVD(n_components=comps, random_state=args.seed)
        item_factors = svd.fit_transform(mat).astype(np.float64)
        norms = np.linalg.norm(item_factors, axis=1)
        norms[norms == 0.0] = 1.0
        item_factors = item_factors / norms[:, None]
        model = None

    # --- Popularity ranking from train only (asin frequency) ---
    pop_rank_asins = train_fit.groupby("asin").size().sort_values(ascending=False).index.tolist()

    use_rw = args.rating_weighted_merge

    def rec_knn(uid, hist_idx: list[int], _held: int, train_asins: list[str]) -> list[int]:
        pairs = [(a, item_to_idx[a]) for a in train_asins if a in item_to_idx]
        if not pairs:
            return []
        hi = [p[1] for p in pairs]
        rw = None
        if use_rw:
            rw = np.array(
                [np.clip(rating_lookup[(uid, p[0])] / 5.0, 0.0, 1.0) for p in pairs],
                dtype=np.float64,
            )
        return _recommend_item_indices(model, mat, hi, args.top_k, row_weights=rw)

    def rec_svd(_uid, hist_idx: list[int], _held: int, train_asins: list[str]) -> list[int]:
        if item_factors is None or not hist_idx:
            return []
        blocked = {item_to_idx[a] for a in train_asins if a in item_to_idx}
        u = item_factors[hist_idx].mean(axis=0)
        un = np.linalg.norm(u)
        if un == 0.0:
            return []
        u = u / un
        scores = item_factors @ u
        if blocked:
            scores[list(blocked)] = -np.inf
        k = min(args.top_k, scores.size - 1)
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])].tolist()

    def rec_popular(_uid, hist_idx: list[int], _held: int, train_asins: list[str]) -> list[int]:
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

    def rec_random(_uid, hist_idx: list[int], _held: int, _train_asins: list[str]) -> list[int]:
        blocked = set(hist_idx)
        pool = [i for i in range(n_items) if i not in blocked]
        rng.shuffle(pool)
        return pool[: args.top_k]

    print(
        f"Model: {args.model} | Split: {args.split} | Implicit matrix: {args.implicit_matrix} | "
        f"User-centered train: {not args.no_center_users and not args.implicit_matrix} | "
        f"BM25: {args.bm25_weighting} (k1={args.bm25_k1}, b={args.bm25_b}) | "
        f"Rating-weighted merge: {use_rw}"
    )

    rec_main = rec_knn if args.model == "knn" else rec_svd
    label = "kNN (item–item)" if args.model == "knn" else f"SVD (n_components={item_factors.shape[1]})"

    h, rr, n = _accumulate_metrics(eval_cases, item_to_idx, args.top_k, rec_main)
    if n == 0:
        raise SystemExit("No evaluation cases; increase data or loosen filters.")

    print(f"\n--- {label} ---")
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
