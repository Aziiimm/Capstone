"""
Offline metrics: temporal hold-out per user → aggregate item–item kNN scores.
Reports HitRate@K, Precision@K (single relevant), NDCG@K.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

import pandas as pd
from sklearn.neighbors import NearestNeighbors

from build_matrix import (
    build_matrix_from_interaction_frame,
    resolve_parquet_paths,
)


def load_reviews_with_time(pattern: str) -> tuple[pd.DataFrame, dict[str, str]]:
    cols_needed = {"reviewerID", "asin", "rating"}
    paths = resolve_parquet_paths(pattern)
    chunks: list[pd.DataFrame] = []
    titles_parts: list[pd.DataFrame] = []
    for path in paths:
        pdf = pd.read_parquet(path)
        missing = cols_needed - set(pdf.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        use = pdf[list(cols_needed)].copy()
        if "timestamp" in pdf.columns:
            use["timestamp"] = pdf["timestamp"]
        else:
            use["timestamp"] = range(len(use))
        chunks.append(use)
        if "product_title" in pdf.columns:
            titles_parts.append(
                pdf[["asin", "product_title"]].drop_duplicates(subset=["asin"])
            )
        else:
            sub = pdf[["asin"]].drop_duplicates(subset=["asin"]).copy()
            sub["product_title"] = sub["asin"].astype(str)
            titles_parts.append(sub)

    df = pd.concat(chunks, ignore_index=True)
    titles = pd.concat(titles_parts, ignore_index=True).drop_duplicates(subset=["asin"])
    titles_by_asin = titles.set_index("asin")["product_title"].astype(str).to_dict()
    df = df.sort_values(["reviewerID", "timestamp"], kind="mergesort")
    return df, titles_by_asin


def temporal_holdout(
    df: pd.DataFrame,
    min_train_interactions: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last interaction per user → test; train is full data minus those rows."""
    test_indices: list[int | pd.Index] = []
    for _, grp in df.groupby("reviewerID", sort=False):
        if len(grp) < min_train_interactions + 1:
            continue
        test_indices.append(grp.index[-1])

    if not test_indices:
        return df.iloc[:0].copy(), df.iloc[:0].copy()

    test_df = df.loc[test_indices].copy()
    train_df = df.drop(test_indices).copy()
    return train_df, test_df


def aggregate_scores_for_user(
    train_item_indices: list[int],
    model: NearestNeighbors,
    X,
    n_neighbors: int,
    exclude: set[int],
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for item_idx in train_item_indices:
        dist, ind = model.kneighbors(X[[item_idx]], n_neighbors=n_neighbors)
        for j, d in zip(ind[0], dist[0]):
            j = int(j)
            if j == item_idx:
                continue
            if j in exclude:
                continue
            scores[j] = scores.get(j, 0.0) + 1.0 / (1.0 + float(d))
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked


def precision_hit_ndcg_at_k(
    ranked_ids: list[int], relevant: int, k: int
) -> tuple[float, float, float]:
    """Single relevant item."""
    top = ranked_ids[:k]
    hit = 1.0 if relevant in top else 0.0
    prec = hit / float(k) if k else 0.0
    if relevant not in top:
        ndcg = 0.0
    else:
        pos = top.index(relevant) + 1  # 1-based
        ndcg = 1.0 / math.log2(pos + 1)  # binary relevance 1 at pos
    return hit, prec, ndcg


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate item–item kNN recommender.")
    ap.add_argument("--path", required=True, help="Parquet glob")
    ap.add_argument("--neighbors", type=int, default=10)
    ap.add_argument("--k-eval", type=int, default=10, help="Cutoff K for metrics")
    ap.add_argument(
        "--rating-mode",
        choices=("raw", "binary"),
        default="raw",
    )
    args = ap.parse_args()

    df, titles_by_asin = load_reviews_with_time(args.path)
    train_df, test_df = temporal_holdout(df, min_train_interactions=1)

    if len(test_df) == 0:
        print("No test users (need >=2 interactions per user).")
        return

    artifacts = build_matrix_from_interaction_frame(
        train_df, titles_by_asin=titles_by_asin, rating_mode=args.rating_mode
    )
    X = artifacts.sparse_csr
    gim = artifacts.global_item_map.set_index("asin")["item_idx"]

    model = NearestNeighbors(
        n_neighbors=args.neighbors,
        metric="cosine",
        algorithm="brute",
    )
    model.fit(X)

    hits = []
    precs = []
    ndcgs = []

    for _, row in test_df.iterrows():
        uid = row["reviewerID"]
        held_asin = row["asin"]
        if held_asin not in gim.index:
            continue
        held_item = int(gim.loc[held_asin])

        user_train = train_df[train_df["reviewerID"] == uid]
        train_items = [
            int(gim.loc[a])
            for a in user_train["asin"].values
            if a in gim.index
        ]
        exclude = set(train_items)
        ranked = aggregate_scores_for_user(
            train_items,
            model,
            X,
            n_neighbors=args.neighbors,
            exclude=exclude,
        )
        ranked_ids = [i for i, _ in ranked]

        h, pr, nd = precision_hit_ndcg_at_k(ranked_ids, held_item, args.k_eval)
        hits.append(h)
        precs.append(pr)
        ndcgs.append(nd)

    n = len(hits)
    if n == 0:
        print("No overlapping test cases after indexing.")
        return

    print("=== Item-item kNN (CPU sklearn, train-only matrix) ===")
    print(f"  test_users_evaluated: {n}")
    print(f"  neighbors: {args.neighbors}  K: {args.k_eval}  rating_mode: {args.rating_mode}")
    print(f"  HitRate@{args.k_eval}: {sum(hits) / n:.4f}")
    print(f"  Precision@{args.k_eval}: {sum(precs) / n:.4f}")
    print(f"  NDCG@{args.k_eval}: {sum(ndcgs) / n:.4f}")


if __name__ == "__main__":
    main()
