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
from content_hybrid import content_scores_from_history, fit_item_title_matrix
from ranking_models import (
    als_dense_scores,
    als_recommend_indices,
    ensemble_recommend_indices,
    fit_als,
    fit_nmf_item_factors,
    fit_svd_item_factors,
    knn_recommend_from_scores,
    knn_score_dict,
    stack_recommend_indices,
    svd_recommend_indices,
)
from sentiment import compound_sentiment


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
    *,
    query_neighbors: int | None = None,
    item_popularity: np.ndarray | None = None,
    popularity_alpha: float = 0.0,
) -> list[int]:
    if not history_item_idx:
        return []
    q = np.asarray(history_item_idx, dtype=np.int64)
    n_query = query_neighbors if query_neighbors is not None else model.n_neighbors
    n_query = min(max(int(n_query), 1), max(1, mat.shape[0] - 1))
    distances, indices = model.kneighbors(mat[q], n_neighbors=n_query)
    if popularity_alpha > 0.0 and item_popularity is not None:
        scores = knn_score_dict(
            model, mat, history_item_idx, row_weights,
            item_popularity=item_popularity, popularity_alpha=popularity_alpha,
            query_neighbors=query_neighbors,
        )
        return knn_recommend_from_scores(scores, top_k)
    return merge_knn_scores(distances, indices, q, top_k, row_weights=row_weights)


def _log_popularity_from_train(train_fit: pd.DataFrame, n_items: int, item_to_idx: dict) -> np.ndarray:
    """log(1 + interaction count) per item index."""
    pop = np.zeros(n_items, dtype=np.float64)
    counts = train_fit.groupby("asin").size()
    for asin, c in counts.items():
        ix = item_to_idx.get(asin)
        if ix is not None:
            pop[ix] = float(np.log1p(c))
    if pop.max() > 0:
        pop /= pop.max()
    return pop


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
        "--sentiment-weighting",
        action="store_true",
        help="Use reviewText sentiment to weight interactions (requires reviewText column).",
    )
    parser.add_argument(
        "--sentiment-alpha",
        type=float,
        default=0.5,
        help="Sentiment strength for weighting: weight = clip(1 + alpha*compound, 0, 2). Default 0.5.",
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
        choices=("knn", "svd", "als", "ensemble", "stack", "nmf"),
        default="knn",
        help="knn | svd | als | ensemble (kNN+SVD) | stack (kNN+SVD+ALS) | nmf.",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=64,
        help="Latent dimension for --model svd (default 64; will be clipped to matrix size).",
    )
    parser.add_argument("--als-factors", type=int, default=64, help="Latent dim for --model als.")
    parser.add_argument("--als-iterations", type=int, default=15, help="ALS training iterations.")
    parser.add_argument("--als-regularization", type=float, default=0.01, help="ALS L2 regularization.")
    parser.add_argument(
        "--ensemble-weight",
        type=float,
        default=0.5,
        help="For --model ensemble: weight on kNN vs SVD (0=all SVD, 1=all kNN).",
    )
    parser.add_argument(
        "--popularity-prior",
        action="store_true",
        help="Add train-set popularity boost to kNN/ensemble candidate scores.",
    )
    parser.add_argument(
        "--popularity-alpha",
        type=float,
        default=0.15,
        help="Strength of popularity prior (default 0.15).",
    )
    parser.add_argument(
        "--query-neighbors",
        type=int,
        default=None,
        help="Neighbors retrieved per history item at query time (default: same as --neighbors).",
    )
    parser.add_argument(
        "--min-history-rating",
        type=float,
        default=None,
        help="Only use train items with rating >= this when building recommendations (e.g. 4.0).",
    )
    parser.add_argument(
        "--recency-half-life-days",
        type=float,
        default=None,
        help="Temporal train only: multiply matrix values by exp decay from last interaction (days).",
    )
    parser.add_argument(
        "--knn-metric",
        choices=("cosine", "euclidean"),
        default="cosine",
        help="Distance metric for item–item kNN (default cosine).",
    )
    parser.add_argument(
        "--als-confidence-alpha",
        type=float,
        default=0.0,
        help="ALS confidence scaling: matrix value -> 1 + alpha*value (default 0).",
    )
    parser.add_argument(
        "--stack-weights",
        type=str,
        default="0.25,0.5,0.25",
        help="For --model stack: knn,svd,als weights (comma-separated, default 0.25,0.5,0.25).",
    )
    parser.add_argument(
        "--nmf-components",
        type=int,
        default=64,
        help="Latent dimension for --model nmf.",
    )
    parser.add_argument(
        "--content-hybrid",
        action="store_true",
        help="Blend TF-IDF product_title similarity (requires product_title in Parquet).",
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        default=0.15,
        help="Weight for content vs collaborative (default 0.15). Used with --content-hybrid.",
    )
    parser.add_argument(
        "--content-max-features",
        type=int,
        default=4000,
        help="Max TF-IDF vocabulary size for title vectors.",
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
    if args.content_hybrid:
        if "product_title" not in cols:
            raise SystemExit(
                "Content hybrid requires 'product_title' in Parquet. "
                "Rebuild pipeline output or disable --content-hybrid."
            )
        read_cols.append("product_title")
    if args.sentiment_weighting:
        if "reviewText" not in cols:
            raise SystemExit(
                "Sentiment weighting requires column 'reviewText' in Parquet. "
                "Rebuild pipeline output with reviewText or disable --sentiment-weighting."
            )
        read_cols.append("reviewText")
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
            row = {"reviewerID": r["reviewerID"], "asin": r["asin"], "rating": float(r["rating"])}
            if args.sentiment_weighting and "reviewText" in r:
                row["reviewText"] = r.get("reviewText")
            train_rows.append(row)
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

    sentiment_lookup: dict[tuple, float] = {}
    if args.sentiment_weighting:
        # reviewText can be missing/NaN; treat as neutral.
        for _, r in train_df.iterrows():
            txt = r.get("reviewText")
            if txt is None or (isinstance(txt, float) and np.isnan(txt)):
                s = 0.0
            else:
                s = compound_sentiment(str(txt))
            sentiment_lookup[(r["reviewerID"], r["asin"])] = float(s)

    def _sent_weight(uid, asin) -> float:
        if not args.sentiment_weighting:
            return 1.0
        s = sentiment_lookup.get((uid, asin), 0.0)
        return float(np.clip(1.0 + args.sentiment_alpha * s, 0.0, 2.0))

    train_fit = train_df.copy()
    if args.sentiment_weighting:
        # Apply sentiment to matrix values before any centering/implicit replacement.
        train_fit["sent_w"] = [
            _sent_weight(uid, a) for uid, a in zip(train_fit["reviewerID"], train_fit["asin"])
        ]
        train_fit["rating"] = train_fit["rating"].astype(np.float32) * train_fit["sent_w"].astype(np.float32)

    if args.implicit_matrix:
        if args.sentiment_weighting:
            # Keep only sentiment weight as implicit value in [0,2]
            train_fit["rating"] = train_fit["sent_w"].astype(np.float32)
        else:
            train_fit["rating"] = 1.0
    elif not args.no_center_users:
        train_fit["rating"] = train_fit.groupby("reviewerID")["rating"].transform(lambda x: x - x.mean())

    if args.recency_half_life_days is not None and args.split == "temporal" and "timestamp" in train_df.columns:
        hl = max(float(args.recency_half_life_days), 1e-6)
        ts_df = train_df[["reviewerID", "asin", "timestamp"]].drop_duplicates(
            subset=["reviewerID", "asin"], keep="last"
        )
        train_fit = train_fit.merge(ts_df, on=["reviewerID", "asin"], how="left")
        ts = pd.to_numeric(train_fit["timestamp"], errors="coerce")
        t_max = float(ts.max()) if ts.notna().any() else 0.0
        rec_w = np.exp(-np.log(2.0) * (t_max - ts.values) / (hl * 86400.0))
        rec_w = np.where(np.isnan(rec_w), 1.0, rec_w)
        train_fit["rating"] = train_fit["rating"].astype(np.float32) * rec_w.astype(np.float32)
        train_fit = train_fit.drop(columns=["timestamp"], errors="ignore")

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

    user_item = mat.T.tocsr()
    item_pop = _log_popularity_from_train(train_fit, n_items, item_to_idx) if args.popularity_prior else None
    pop_alpha = args.popularity_alpha if args.popularity_prior else 0.0
    q_neighbors = args.query_neighbors

    title_matrix: np.ndarray | None = None
    if args.content_hybrid and "product_title" in df.columns:
        title_by_asin = (
            df.dropna(subset=["asin"])
            .groupby("asin", as_index=False)["product_title"]
            .first()
            .set_index("asin")["product_title"]
            .to_dict()
        )
        idx_to_asin = {i: a for a, i in item_to_idx.items()}
        titles_by_idx = [
            str(title_by_asin.get(idx_to_asin.get(i, ""), "") or "") for i in range(n_items)
        ]
        title_matrix = fit_item_title_matrix(
            titles_by_idx, max_features=args.content_max_features, seed=args.seed
        )

    model: NearestNeighbors | None = None
    item_factors: np.ndarray | None = None
    nmf_factors: np.ndarray | None = None
    als_model = None

    need_knn = args.model in ("knn", "ensemble", "stack")
    need_svd = args.model in ("svd", "ensemble", "stack")
    need_als = args.model in ("als", "stack")

    if need_knn:
        k_fit = min(max(args.neighbors, 2), max(2, n_items))
        model = NearestNeighbors(n_neighbors=k_fit, metric=args.knn_metric)
        model.fit(mat)
    if need_svd:
        item_factors = fit_svd_item_factors(mat, args.svd_components, args.seed)
    if args.model == "nmf":
        nmf_factors = fit_nmf_item_factors(mat, args.nmf_components, args.seed)
    if need_als:
        als_model = fit_als(
            user_item,
            factors=args.als_factors,
            regularization=args.als_regularization,
            iterations=args.als_iterations,
            seed=args.seed,
            confidence_alpha=args.als_confidence_alpha,
        )

    stack_w = [float(x.strip()) for x in args.stack_weights.split(",")]
    if len(stack_w) != 3:
        raise SystemExit("--stack-weights must have exactly 3 comma-separated values (knn,svd,als).")

    # --- Popularity ranking from train only (asin frequency) ---
    pop_rank_asins = train_fit.groupby("asin").size().sort_values(ascending=False).index.tolist()

    use_rw = args.rating_weighted_merge
    min_hist = args.min_history_rating

    def _filter_train_asins(uid: str, train_asins: list[str]) -> list[str]:
        if min_hist is None:
            return train_asins
        return [a for a in train_asins if rating_lookup.get((uid, a), 0.0) >= min_hist]

    def _history_pairs(uid: str, train_asins: list[str]) -> tuple[list[str], list[int], np.ndarray | None]:
        filtered = _filter_train_asins(uid, train_asins)
        pairs = [(a, item_to_idx[a]) for a in filtered if a in item_to_idx]
        if not pairs:
            return filtered, [], None
        hi = [p[1] for p in pairs]
        rw = None
        if use_rw or args.sentiment_weighting:
            weights = []
            for asin, _ in pairs:
                w = 1.0
                if use_rw:
                    w *= float(np.clip(rating_lookup[(uid, asin)] / 5.0, 0.0, 1.0))
                if args.sentiment_weighting:
                    w *= _sent_weight(uid, asin)
                weights.append(w)
            rw = np.asarray(weights, dtype=np.float64)
        return filtered, hi, rw

    def rec_knn(uid, hist_idx: list[int], _held: int, train_asins: list[str]) -> list[int]:
        if model is None:
            return []
        _, hi, rw = _history_pairs(uid, train_asins)
        if not hi:
            return []
        return _recommend_item_indices(
            model, mat, hi, args.top_k, rw,
            query_neighbors=q_neighbors,
            item_popularity=item_pop,
            popularity_alpha=pop_alpha,
        )

    def rec_svd(uid, hist_idx: list[int], _held: int, train_asins: list[str]) -> list[int]:
        if item_factors is None:
            return []
        _, hi, rw = _history_pairs(uid, train_asins)
        if not hi:
            return []
        blocked = {item_to_idx[a] for a in train_asins if a in item_to_idx}
        return svd_recommend_indices(item_factors, hi, blocked, args.top_k, rw)

    def rec_als(uid, hist_idx: list[int], _held: int, train_asins: list[str]) -> list[int]:
        if als_model is None:
            return []
        uix = user_to_idx.get(uid)
        if uix is None:
            return []
        blocked = {item_to_idx[a] for a in train_asins if a in item_to_idx}
        row = user_item[uix]
        return als_recommend_indices(als_model, uix, blocked, args.top_k, row)

    def _svd_user_scores(hi: list[int], rw: np.ndarray | None) -> np.ndarray:
        if item_factors is None or not hi:
            return np.zeros(n_items, dtype=np.float64)
        if rw is not None and rw.sum() > 0:
            u = (item_factors[hi] * rw[:, None]).sum(axis=0) / rw.sum()
        else:
            u = item_factors[hi].mean(axis=0)
        un = np.linalg.norm(u)
        if un > 0:
            u = u / un
        return item_factors @ u

    def rec_ensemble(uid, hist_idx: list[int], _held: int, train_asins: list[str]) -> list[int]:
        if model is None or item_factors is None:
            return []
        _, hi, rw = _history_pairs(uid, train_asins)
        if not hi:
            return []
        blocked = {item_to_idx[a] for a in train_asins if a in item_to_idx}
        knn_sc = knn_score_dict(
            model, mat, hi, rw,
            item_popularity=item_pop, popularity_alpha=pop_alpha,
            query_neighbors=q_neighbors,
        )
        svd_sc = _svd_user_scores(hi, rw)
        if args.content_hybrid and title_matrix is not None:
            cw = args.content_weight
            cf_w = 1.0 - cw
            knn_w = args.ensemble_weight * cf_w
            svd_w = (1.0 - args.ensemble_weight) * cf_w
            content_sc = content_scores_from_history(title_matrix, hi, rw, blocked)
            return stack_recommend_indices(
                [(knn_w, knn_sc), (svd_w, svd_sc), (cw, content_sc)],
                blocked, args.top_k, n_items,
            )
        return ensemble_recommend_indices(
            knn_sc, svd_sc, blocked, args.top_k, knn_weight=args.ensemble_weight,
        )

    def rec_stack(uid, hist_idx: list[int], _held: int, train_asins: list[str]) -> list[int]:
        if model is None or item_factors is None or als_model is None:
            return []
        _, hi, rw = _history_pairs(uid, train_asins)
        if not hi:
            return []
        blocked = {item_to_idx[a] for a in train_asins if a in item_to_idx}
        knn_sc = knn_score_dict(
            model, mat, hi, rw,
            item_popularity=item_pop, popularity_alpha=pop_alpha,
            query_neighbors=q_neighbors,
        )
        svd_sc = _svd_user_scores(hi, rw)
        uix = user_to_idx.get(uid)
        als_sc = (
            als_dense_scores(als_model, uix, n_items)
            if uix is not None
            else np.zeros(n_items, dtype=np.float64)
        )
        sources: list[tuple[float, dict[int, float] | np.ndarray]] = [
            (stack_w[0], knn_sc),
            (stack_w[1], svd_sc),
            (stack_w[2], als_sc),
        ]
        if args.content_hybrid and title_matrix is not None:
            content_sc = content_scores_from_history(title_matrix, hi, rw, blocked)
            sources.append((args.content_weight, content_sc))
        return stack_recommend_indices(sources, blocked, args.top_k, n_items)

    def rec_nmf(uid, hist_idx: list[int], _held: int, train_asins: list[str]) -> list[int]:
        if nmf_factors is None:
            return []
        _, hi, rw = _history_pairs(uid, train_asins)
        if not hi:
            return []
        blocked = {item_to_idx[a] for a in train_asins if a in item_to_idx}
        return svd_recommend_indices(nmf_factors, hi, blocked, args.top_k, rw)

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
        f"Rating-weighted merge: {use_rw} | Sentiment: {args.sentiment_weighting} (alpha={args.sentiment_alpha}) | "
        f"Pop prior: {args.popularity_prior} (alpha={args.popularity_alpha}) | "
        f"Min history rating: {min_hist} | Recency half-life (days): {args.recency_half_life_days} | "
        f"kNN metric: {args.knn_metric} | Content hybrid: {args.content_hybrid} (w={args.content_weight})"
    )

    if args.model == "knn":
        rec_main = rec_knn
        label = f"kNN (metric={args.knn_metric})"
    elif args.model == "svd":
        rec_main = rec_svd
        label = f"SVD (n_components={item_factors.shape[1] if item_factors is not None else 0})"
    elif args.model == "als":
        rec_main = rec_als
        label = f"ALS (factors={args.als_factors}, iter={args.als_iterations}, alpha={args.als_confidence_alpha})"
    elif args.model == "nmf":
        rec_main = rec_nmf
        label = f"NMF (n_components={nmf_factors.shape[1] if nmf_factors is not None else 0})"
    elif args.model == "stack":
        rec_main = rec_stack
        label = f"Stack kNN+SVD+ALS (weights={stack_w})"
    else:
        rec_main = rec_ensemble
        label = f"Ensemble kNN+SVD (knn_weight={args.ensemble_weight})"

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
