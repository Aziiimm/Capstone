"""
Load review data (and optional item metadata) from local JSONL.GZ (UCSD datarepo).
With one file: outputs core columns only (reviewerID, asin, rating, reviewText, timestamp).
With two files (review + meta): joins on parent_asin and adds product_title, main_category.
Uses Dask (or Dask-cuDF when use_gpu=True) for out-of-core read; no full load into memory.
"""
from __future__ import annotations

import os
from typing import Union
import pandas as pd

try:
    import dask_cudf
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

import dask.dataframe as dd

# Schema configuration
CORE_COLUMNS = ["reviewerID", "asin", "rating", "reviewText", "timestamp"]
META_COLUMNS = ["product_title", "main_category"]
OUTPUT_COLUMNS_JOINED = CORE_COLUMNS + META_COLUMNS

REVIEW_COLUMN_MAP = {
    "user_id": "reviewerID",
    "text": "reviewText",
    "asin": "asin",
    "rating": "rating",
    "timestamp": "timestamp",
}
JOIN_KEY = "parent_asin"

def _load_review(path: str, blocksize: str, use_gpu: bool = False) -> Union[dd.DataFrame, dask_cudf.DataFrame]:
    """Load review JSONL natively on GPU if use_gpu is True."""
    if use_gpu and HAS_CUDF:
        dtype_map = {
            "user_id": "str", 
            "asin": "str", 
            "text": "str", 
            "rating": "float32"
        }
        
        ddf = dask_cudf.read_json(
            path, 
            lines=True, 
            blocksize=blocksize,
            compression=None,
            dtype=dtype_map 
        )
    else:
        # Standard CPU path
        ddf = dd.read_json(path, lines=True, blocksize=blocksize)

    # 1. Rename columns immediately
    ddf = ddf.rename(columns=REVIEW_COLUMN_MAP)
    
    # 2. Ensure JOIN_KEY exists (mapping asin to parent_asin if needed)
    if JOIN_KEY not in ddf.columns and "asin" in ddf.columns:
        ddf = ddf.assign(**{JOIN_KEY: ddf["asin"]})
        
    # 3. Strictly filter to only the columns that the rest of your pipeline expects
    valid_cols = [c for c in CORE_COLUMNS + [JOIN_KEY] if c in ddf.columns]
    return ddf[valid_cols]

def _load_meta(path: str, use_gpu: bool = False) -> Union[dd.DataFrame, dask_cudf.DataFrame]:
    """Load metadata on GPU with explicit string dtypes to prevent schema mismatches."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Meta file not found: {path}")

    if use_gpu and HAS_CUDF:
        messy_columns = {
            "author": "object", 
            "details": "object", 
            "images": "object", 
            "video_360": "object",
            "price": "object",  
            "videos": "object",  
            "feature": "object",
            "description": "object"
        }
        
        # We use dask_cudf.read_json directly for partitioned loading
        df = dask_cudf.read_json(
            path, 
            lines=True, 
            blocksize="128MB", 
            dtype=messy_columns,
            compression=None
        )
    else:
        # Standard CPU path consistency
        df = dd.read_json(path, lines=True, blocksize="128MB")

    # Standardize the Join Key
    if "parent_asin" in df.columns:
        df = df.rename(columns={"parent_asin": JOIN_KEY})
    
    # Rename and Filter to only the 3 columns we actually need
    df = df.rename(columns={"title": "product_title", "main_category": "main_category"})
    keep_cols = [c for c in [JOIN_KEY, "product_title", "main_category"] if c in df.columns]
    
    # Drop junk columns IMMEDIATELY to free up VRAM on your 3060 Ti
    df = df[keep_cols]
    
    # Final cleanup logic
    df = df.drop_duplicates(subset=[JOIN_KEY])
    df[JOIN_KEY] = df[JOIN_KEY].astype(str)

    return df

def _join_review_meta(ddf_review: dd.DataFrame, ddf_meta: dd.DataFrame) -> dd.DataFrame:
    """Perform GPU-accelerated merge."""
    ddf_review = ddf_review.assign(**{JOIN_KEY: ddf_review[JOIN_KEY].astype(str)})
    ddf_meta = ddf_meta.assign(**{JOIN_KEY: ddf_meta[JOIN_KEY].astype(str)})
    
    # Left merge: keep all reviews, add meta info where available
    merged = ddf_review.merge(ddf_meta, on=JOIN_KEY, how="left")
    
    # Ensure reviewerID and asin are definitely kept for filtering
    required_cols = ["reviewerID", "asin", "rating", "reviewText", "timestamp", "product_title", "main_category"]
    out_cols = [c for c in required_cols if c in merged.columns]

    # Maintain GPU-native structure
    if HAS_CUDF and isinstance(merged, dask_cudf.DataFrame):
        meta = cudf.DataFrame(columns=out_cols)
    else:
        meta = pd.DataFrame(columns=out_cols)

    return merged.map_partitions(lambda part: part[out_cols].copy(), meta=merged._meta[out_cols])

def load_from_local(
    path: str,
    meta_path: str | None = None,
    limit: int | None = None,
    use_gpu: bool = False,
    blocksize: str = "64MB",
) -> Union[dd.DataFrame, dask_cudf.DataFrame]:
    
    if use_gpu and not HAS_CUDF:
        use_gpu = False

    # Load reviews (and meta if provided)
    ddf_review = _load_review(path, blocksize, use_gpu=use_gpu)
    
    if meta_path:
        ddf_meta = _load_meta(meta_path, use_gpu=use_gpu)
        ddf = _join_review_meta(ddf_review, ddf_meta)
    else:
        ddf = ddf_review[CORE_COLUMNS]

    if limit is not None:
        ddf = ddf.head(limit)

    return ddf


def load(
    source: str,
    meta: str | None = None,
    limit: int | None = None,
    use_gpu: bool = False,
    **kwargs,
) -> Union[dd.DataFrame, "dask_cudf.DataFrame"]:
    """
    Load review data (and optional item metadata).
    - source: path to review JSONL.GZ.
    - meta: optional path to meta JSONL.GZ (same category). If set, join and add product_title, main_category.
    """
    def _resolve_file(path_arg: str, label: str) -> str:
        resolved = os.path.abspath(os.path.expanduser(path_arg))
        if os.path.isfile(resolved):
            return resolved
        cwd = os.getcwd()
        basename = os.path.basename(path_arg)
        # Try exact folder names
        for folder in ("dataset", "downloads", "data", "."):
            candidate = os.path.join(cwd, folder, basename) if folder != "." else os.path.join(cwd, basename)
            if os.path.isfile(candidate):
                return candidate
        # Build helpful error: list any .jsonl.gz files found under cwd
        found = []
        for folder in ("dataset", "downloads", "data", ".", ""):
            try:
                dirpath = os.path.join(cwd, folder) if folder else cwd
                if not os.path.isdir(dirpath) and folder:
                    continue
                for name in os.listdir(dirpath):
                    if name.endswith(".jsonl.gz"):
                        found.append(os.path.join(folder, name) if folder and folder != "." else name)
            except OSError:
                pass
        hint = f" Found .jsonl.gz here: {found}." if found else " No .jsonl.gz files found in dataset/, downloads/, or data/."
        raise ValueError(
            f"{label} not found: {resolved}. "
            f"Tried dataset/, downloads/, data/.{hint} "
            "Put the file there or pass an absolute path. See README for download URLs."
        )

    source = _resolve_file(source, "Source file")
    meta_resolved = _resolve_file(meta, "Meta file") if meta else None
    return load_from_local(
        source,
        meta_path=meta_resolved,
        limit=limit,
        use_gpu=use_gpu,
        **kwargs,
    )
