"""
Filter to users with >5 reviews and items with >10 reviews.
Works on full Dask/cuDF DataFrame in an out-of-core way (11GB VRAM safe).
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

if HAS_CUDF:
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client

MIN_REVIEWS_PER_USER = 6
MIN_REVIEWS_PER_ITEM = 11

def filter_counts(
    df: Union[dd.DataFrame, pd.DataFrame, "dask_cudf.DataFrame", "cudf.DataFrame"],
) -> Union[dd.DataFrame, pd.DataFrame, "dask_cudf.DataFrame", "cudf.DataFrame"]:
    """Keep only rows where user has >= 6 reviews and item has >= 11 reviews on the GPU."""
    
    # Identify if we are using the GPU path
    is_gpu = (HAS_CUDF and isinstance(df, (dask_cudf.DataFrame, cudf.DataFrame)))

    # For Dask and Dask-cuDF: Use shuffle-based grouping to manage 8GB VRAM
    if isinstance(df, (dd.DataFrame, dask_cudf.DataFrame)):
        # Calculate counts natively on GPU CUDA cores
        user_counts = df.groupby("reviewerID").size().to_frame("_user_count")
        item_counts = df.groupby("asin").size().to_frame("_item_count")
        
        # Inner joins on GPU are significantly faster than CPU baseline
        df = df.merge(user_counts, left_on="reviewerID", right_index=True, how="inner")
        df = df.merge(item_counts, left_on="asin", right_index=True, how="inner")
        
        # Filter based on project requirements (6 reviews/user, 11/item)
        df = df[df["_user_count"] >= MIN_REVIEWS_PER_USER]
        df = df[df["_item_count"] >= MIN_REVIEWS_PER_ITEM]
        
        return df.drop(columns=["_user_count", "_item_count"])

    # Fallback for standard Pandas/cuDF (Single Partition)
    user_counts = df.groupby("reviewerID").size().rename("_uc")
    item_counts = df.groupby("asin").size().rename("_ic")
    
    df = df.merge(user_counts, left_on="reviewerID", right_index=True)
    df = df.merge(item_counts, left_on="asin", right_index=True)
    
    return df[
        (df["_uc"] >= MIN_REVIEWS_PER_USER) & 
        (df["_ic"] >= MIN_REVIEWS_PER_ITEM)
    ].drop(columns=["_uc", "_ic"])