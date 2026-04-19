"""
Write filtered DataFrame to Parquet (single file or directory of parts).
"""
from __future__ import annotations
from typing import Union
import dask.dataframe as dd
try:
    import dask_cudf
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

# The 7 specific columns required for the recommender service
FINAL_COLUMNS = [
    "reviewerID", 
    "asin", 
    "rating", 
    "reviewText", 
    "timestamp", 
    "product_title", 
    "main_category"
]

def to_parquet(
    df: Union[dd.DataFrame, "dask_cudf.DataFrame"],
    path: str,
    **kwargs,
) -> None:
    # 1. Selection: This drops 'parent_asin' and ensures exactly 7 columns
    df = df[FINAL_COLUMNS]
    
    # 2. Cleanup: Reset index to ensure a clean start
    df = df.reset_index(drop=True)
    
    # 3. Writing: write_index=False removes the '__null_dask_index__'
    df.to_parquet(path, write_index=False, **kwargs)