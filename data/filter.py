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


MIN_REVIEWS_PER_USER = 6
MIN_REVIEWS_PER_ITEM = 11


def filter_counts(
    df: Union[dd.DataFrame, pd.DataFrame, "dask_cudf.DataFrame", "cudf.DataFrame"],
) -> Union[dd.DataFrame, pd.DataFrame, "dask_cudf.DataFrame", "cudf.DataFrame"]:
    """Keep only rows where user has >= 6 reviews and item has >= 11 reviews."""
    is_dask = isinstance(df, dd.DataFrame)
    is_dask_cudf = HAS_CUDF and isinstance(df, dask_cudf.DataFrame)

    if is_dask or is_dask_cudf:
        user_counts = df.groupby("reviewerID").size().rename("_user_count").to_frame()
        item_counts = df.groupby("asin").size().rename("_item_count").to_frame()
        df = df.merge(
            user_counts,
            left_on="reviewerID",
            right_index=True,
            how="inner",
        )
        df = df.merge(
            item_counts,
            left_on="asin",
            right_index=True,
            how="inner",
        )
        df = df[df["_user_count"] >= MIN_REVIEWS_PER_USER]
        df = df[df["_item_count"] >= MIN_REVIEWS_PER_ITEM]
        df = df.drop(columns=["_user_count", "_item_count"])
        return df

    user_counts = df.groupby("reviewerID").size().rename("_uc")
    item_counts = df.groupby("asin").size().rename("_ic")
    df = df.merge(user_counts, left_on="reviewerID", right_index=True)
    df = df.merge(item_counts, left_on="asin", right_index=True)
    df = df[df["_uc"] >= MIN_REVIEWS_PER_USER]
    df = df[df["_ic"] >= MIN_REVIEWS_PER_ITEM]
    return df.drop(columns=["_uc", "_ic"])
