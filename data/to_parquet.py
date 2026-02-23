"""
Write filtered DataFrame to Parquet (single file or directory of parts).
"""
from __future__ import annotations

from typing import Union

import dask.dataframe as dd

try:
    import dask_cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

def to_parquet(
    df: Union[dd.DataFrame, "dask_cudf.DataFrame"],
    path: str,
    **kwargs,
) -> None:
    """
    Write Dask or Dask-cuDF DataFrame to Parquet using the best available engine.
    """
    # Reset index so we don't write it; avoids PyArrow write_index compat issues
    df = df.reset_index(drop=True)

    # Use the GPU engine if we are dealing with a dask_cudf object
    if HAS_CUDF and isinstance(df, dask_cudf.DataFrame):
        # engine='cudf' writes natively from VRAM to disk
        df.to_parquet(path, engine="cudf", **kwargs)
    else:
        # Standard CPU-based writing for teammates or CPU-only runs
        df.to_parquet(path, **kwargs)