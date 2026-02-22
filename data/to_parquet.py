"""
Write filtered DataFrame to Parquet (single file or directory of parts).
"""
from __future__ import annotations

from typing import Union

import dask.dataframe as dd


def to_parquet(
    df: Union[dd.DataFrame, "dask_cudf.DataFrame"],
    path: str,
    **kwargs,
) -> None:
    """
    Write Dask or Dask-cuDF DataFrame to Parquet.
    path can be a directory (partitioned) or a single file path.
    """
    # Reset index so we don't write it; avoids PyArrow write_index compat issues
    df = df.reset_index(drop=True)
    df.to_parquet(path, **kwargs)
