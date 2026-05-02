"""
Filter to users with >5 reviews and items with >10 reviews.
Optimized for GPU (dask-cudf) by persisting intermediate join results.
"""
from __future__ import annotations
from typing import Union
import pandas as pd

try:
    import dask_cudf
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

import dask.dataframe as dd

# Thresholds per senior project requirements
MIN_REVIEWS_PER_USER = 6
MIN_REVIEWS_PER_ITEM = 11

def filter_counts(
    df: Union[dd.DataFrame, pd.DataFrame, "dask_cudf.DataFrame", "cudf.DataFrame"],
    min_reviews_per_user: int | None = None,
    min_reviews_per_item: int | None = None,
) -> Union[dd.DataFrame, pd.DataFrame, "dask_cudf.DataFrame", "cudf.DataFrame"]:
    """Keep only rows where user/item meet minimum review counts (defaults: project thresholds)."""
    min_u = MIN_REVIEWS_PER_USER if min_reviews_per_user is None else min_reviews_per_user
    min_i = MIN_REVIEWS_PER_ITEM if min_reviews_per_item is None else min_reviews_per_item
    is_dask = isinstance(df, dd.DataFrame)

    if is_dask:
            # 1. Calculate counts
            user_counts = df.groupby("reviewerID").size().to_frame("_user_count")
            item_counts = df.groupby("asin").size().to_frame("_item_count")
            
            # 2. Join (Inner joins are memory-heavy on GPU)
            df = df.merge(user_counts, left_on="reviewerID", right_index=True, how="inner")
            df = df.merge(item_counts, left_on="asin", right_index=True, how="inner")
            
            # 3. Filter
            df = df[(df["_user_count"] >= min_u) & (df["_item_count"] >= min_i)]
            
            df = df.drop(columns=["_user_count", "_item_count"])
            
            # If we have millions of rows, skipping persist saves the GPU from crashing.
            if df.npartitions < 5: 
                return df.persist()
            return df # Stream directly to disk for large datasets
    
    # Standard Pandas/cuDF path for smaller, in-memory datasets
    user_counts = df.groupby("reviewerID").size().rename("_uc")
    item_counts = df.groupby("asin").size().rename("_ic")
    df = df.merge(user_counts, left_on="reviewerID", right_index=True)
    df = df.merge(item_counts, left_on="asin", right_index=True)
    df = df[df["_uc"] >= min_u]
    df = df[df["_ic"] >= min_i]
    return df.drop(columns=["_uc", "_ic"])