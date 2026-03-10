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
    # 1. Selection: Drops 'parent_asin' and ensures exactly 7 columns
    df = df[FINAL_COLUMNS]
    
    # 2. Cleanup: Reset index to ensure a clean start
    df = df.reset_index(drop=True)
    
    # 3. IVAN'S FIX: Cast object-dtype columns to str so PyArrow can infer the schema correctly.
    # Without this, Dask's _meta_nonempty fills object columns with bare Python
    # `object()` sentinel values that PyArrow cannot convert.
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        df = df.assign(**{c: df[c].astype(str) for c in obj_cols})
        
    # 4. Writing: write_index=False removes the '__null_dask_index__'
    df.to_parquet(path, write_index=False, **kwargs)
