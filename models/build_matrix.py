"""
Load Parquet review data and build the item × user CSR matrix (CPU / SciPy).
Shared by CPU training, GPU training (upload to GPU), benchmarking, and evaluation.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class MatrixArtifacts:
    sparse_csr: sparse.csr_matrix
    title_map: dict[int, str]
    n_users: int
    n_items: int
    nnz: int
    global_user_map: pd.DataFrame  # columns reviewerID, user_idx
    global_item_map: pd.DataFrame  # columns asin, item_idx


def resolve_parquet_paths(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern!r}")
    return paths


def load_parquet_interactions(
    pattern: str,
    min_reviews_per_user: int = 1,
    min_reviews_per_item: int = 1,
    rating_mode: str = "raw",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """
    Load all Parquet files matching pattern.

    rating_mode:
      - \"raw\": use rating column as-is
      - \"binary\": 1.0 for any positive rating (implicit-style)
    """
    paths = resolve_parquet_paths(pattern)
    parts: list[pd.DataFrame] = []
    title_chunks: list[pd.DataFrame] = []

    for p in paths:
        df = pd.read_parquet(p, columns=["reviewerID", "asin", "rating"])
        parts.append(df)
        meta = pd.read_parquet(p)
        if "product_title" in meta.columns:
            title_chunks.append(meta[["asin", "product_title"]].drop_duplicates(subset=["asin"]))
        else:
            title_chunks.append(meta[["asin"]].assign(product_title=meta["asin"].astype(str)))

    df_all = pd.concat(parts, ignore_index=True)

    if min_reviews_per_user > 1:
        uc = df_all.groupby("reviewerID").size()
        keep_u = uc[uc >= min_reviews_per_user].index
        df_all = df_all[df_all["reviewerID"].isin(keep_u)]

    if min_reviews_per_item > 1:
        ic = df_all.groupby("asin").size()
        keep_i = ic[ic >= min_reviews_per_item].index
        df_all = df_all[df_all["asin"].isin(keep_i)]

    users = pd.unique(df_all["reviewerID"])
    items = pd.unique(df_all["asin"])

    global_user_map = pd.DataFrame(
        {"reviewerID": users, "user_idx": np.arange(len(users), dtype=np.int64)}
    )
    global_item_map = pd.DataFrame(
        {"asin": items, "item_idx": np.arange(len(items), dtype=np.int64)}
    )

    df = df_all.merge(global_user_map, on="reviewerID", how="inner").merge(
        global_item_map, on="asin", how="inner"
    )

    if rating_mode == "binary":
        df = df.assign(rating=np.where(df["rating"].astype(float) > 0, 1.0, 0.0))
    else:
        df = df.assign(rating=df["rating"].astype(np.float32))

    full_titles = pd.concat(title_chunks, ignore_index=True).drop_duplicates(subset=["asin"])
    final_mapping = full_titles.merge(global_item_map, on="asin", how="inner")
    title_map = final_mapping.set_index("item_idx")["product_title"].astype(str).to_dict()

    return df, global_user_map, global_item_map, title_map


def interactions_to_csr(
    df: pd.DataFrame, n_items: int, n_users: int
) -> sparse.csr_matrix:
    rows = df["item_idx"].to_numpy(dtype=np.int64)
    cols = df["user_idx"].to_numpy(dtype=np.int64)
    data = df["rating"].to_numpy(dtype=np.float64)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=(n_items, n_users)).tocsr()
    mat.eliminate_zeros()
    return mat


def build_matrix_from_parquet(
    pattern: str,
    min_reviews_per_user: int = 1,
    min_reviews_per_item: int = 1,
    rating_mode: str = "raw",
) -> MatrixArtifacts:
    df, gum, gim, title_map = load_parquet_interactions(
        pattern,
        min_reviews_per_user=min_reviews_per_user,
        min_reviews_per_item=min_reviews_per_item,
        rating_mode=rating_mode,
    )
    n_users = len(gum)
    n_items = len(gim)
    csr = interactions_to_csr(df, n_items=n_items, n_users=n_users)
    nnz = csr.nnz
    return MatrixArtifacts(
        sparse_csr=csr,
        title_map=title_map,
        n_users=n_users,
        n_items=n_items,
        nnz=nnz,
        global_user_map=gum,
        global_item_map=gim,
    )


def dataframe_to_artifacts(
    df: pd.DataFrame,
    title_map: dict[int, str],
    global_user_map: pd.DataFrame,
    global_item_map: pd.DataFrame,
) -> MatrixArtifacts:
    """Build CSR from an already-indexed dataframe (same columns as load output)."""
    n_users = len(global_user_map)
    n_items = len(global_item_map)
    csr = interactions_to_csr(df, n_items=n_items, n_users=n_users)
    return MatrixArtifacts(
        sparse_csr=csr,
        title_map=title_map,
        n_users=n_users,
        n_items=n_items,
        nnz=csr.nnz,
        global_user_map=global_user_map,
        global_item_map=global_item_map,
    )


def build_matrix_from_interaction_frame(
    df: pd.DataFrame,
    titles_by_asin: dict[str, str] | None = None,
    rating_mode: str = "raw",
) -> MatrixArtifacts:
    """
    Build artifacts from a reviews dataframe (reviewerID, asin, rating).
    Used for evaluation trains on a filtered subset.
    """
    df_work = df[["reviewerID", "asin", "rating"]].copy()
    users = pd.unique(df_work["reviewerID"])
    items = pd.unique(df_work["asin"])
    global_user_map = pd.DataFrame(
        {"reviewerID": users, "user_idx": np.arange(len(users), dtype=np.int64)}
    )
    global_item_map = pd.DataFrame(
        {"asin": items, "item_idx": np.arange(len(items), dtype=np.int64)}
    )
    merged = df_work.merge(global_user_map, on="reviewerID").merge(
        global_item_map, on="asin"
    )
    if rating_mode == "binary":
        merged = merged.assign(
            rating=np.where(merged["rating"].astype(float) > 0, 1.0, 0.0)
        )
    else:
        merged = merged.assign(rating=merged["rating"].astype(np.float32))

    title_map = {}
    for row in global_item_map.itertuples(index=False):
        asin_s = str(row.asin)
        if titles_by_asin is None:
            title_map[int(row.item_idx)] = asin_s
        else:
            title_map[int(row.item_idx)] = titles_by_asin.get(asin_s, asin_s)

    return dataframe_to_artifacts(
        merged, title_map, global_user_map, global_item_map
    )
