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


# Core columns required for filter and downstream
CORE_COLUMNS = ["reviewerID", "asin", "rating", "reviewText", "timestamp"]
# Meta columns added when --meta file is provided (join with item metadata)
META_COLUMNS = ["product_title", "main_category"]
REQUIRED_COLUMNS = CORE_COLUMNS  # filter expects at least these
OUTPUT_COLUMNS_JOINED = CORE_COLUMNS + META_COLUMNS

# UCSD/Hugging Face raw review fields -> our schema
REVIEW_COLUMN_MAP = {
    "user_id": "reviewerID",
    "text": "reviewText",
    "asin": "asin",
    "rating": "rating",
    "timestamp": "timestamp",
}
# Raw meta fields -> our schema (only those we keep)
META_COLUMN_MAP = {
    "title": "product_title",
    "main_category": "main_category",
}
JOIN_KEY = "parent_asin"


def _load_review(path: str, blocksize: str) -> dd.DataFrame:
    """Load review JSONL; return Dask DataFrame with core columns + parent_asin for join."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Review file not found: {path}. "
            "Download from the UCSD datarepo (see README)."
        )
    ddf = dd.read_json(path, lines=True, blocksize=blocksize)
    # Map raw names to our schema
    rename = {k: v for k, v in REVIEW_COLUMN_MAP.items() if k in ddf.columns}
    ddf = ddf.rename(columns=rename)
    # We need parent_asin for join; keep it until after join
    need = [c for c in CORE_COLUMNS if c in ddf.columns] + [JOIN_KEY]
    missing = [c for c in CORE_COLUMNS if c not in ddf.columns]
    if missing:
        raise ValueError(
            f"Review file missing columns {missing}. Expected after mapping: {CORE_COLUMNS}. "
            f"File has: {list(ddf.columns)}"
        )
    # Ensure parent_asin exists (required for join)
    if JOIN_KEY not in ddf.columns and "parent_asin" in ddf.columns:
        pass  # already there with raw name
    elif "parent_asin" in ddf.columns and JOIN_KEY not in ddf.columns:
        ddf = ddf.rename(columns={"parent_asin": JOIN_KEY})
    else:
        # Fallback: use asin as parent_asin if dataset has no parent_asin
        if JOIN_KEY not in ddf.columns:
            ddf = ddf.assign(**{JOIN_KEY: ddf["asin"]})
    return ddf[[c for c in ddf.columns if c in CORE_COLUMNS + [JOIN_KEY]]]


def _load_meta(path: str, blocksize: str) -> dd.DataFrame:
    """
    Load item metadata JSONL lazily using Dask.
    Replaced pandas read_json to prevent massive task graph serialization on 5GB+ files.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Meta file not found: {path}. "
            "Download from the UCSD datarepo (see README)."
        )
    
    # FIX: Use Dask's lazy reader instead of Pandas
    ddf = dd.read_json(path, lines=True, blocksize=blocksize)
    
    rename = {"title": "product_title", "main_category": "main_category"}
    
    if "parent_asin" in ddf.columns:
        ddf = ddf.rename(columns={"parent_asin": JOIN_KEY})
        
    for raw, out in rename.items():
        if raw in ddf.columns:
            ddf = ddf.rename(columns={raw: out})
            
    keep = [c for c in [JOIN_KEY, "product_title", "main_category"] if c in ddf.columns]
    
    if JOIN_KEY not in keep:
        raise ValueError(f"Meta file missing '{JOIN_KEY}'. Has: {list(ddf.columns)}")
        
    # Drop duplicates and enforce string type lazily
    ddf = ddf[keep].drop_duplicates(subset=[JOIN_KEY])
    ddf[JOIN_KEY] = ddf[JOIN_KEY].astype(str)
    
    return ddf


def _join_partition_keep_columns(part: pd.DataFrame, out_cols: list[str]) -> pd.DataFrame:
    """Keep only output columns; force clean schema to avoid metadata mismatch."""
    keep = [c for c in out_cols if c in part.columns]
    return part[keep].copy()


def _join_review_meta(
    ddf_review: dd.DataFrame,
    ddf_meta: dd.DataFrame,
) -> dd.DataFrame:
    """Left-join review with meta on parent_asin; return DataFrame with OUTPUT_COLUMNS_JOINED."""
    # Cast join key to string on both sides to avoid dtype mismatch
    ddf_review = ddf_review.assign(**{JOIN_KEY: ddf_review[JOIN_KEY].astype(str)})
    ddf_meta = ddf_meta.assign(**{JOIN_KEY: ddf_meta[JOIN_KEY].astype(str)})
    merged = ddf_review.merge(ddf_meta, on=JOIN_KEY, how="left")
    out_cols = [c for c in OUTPUT_COLUMNS_JOINED if c in merged.columns]
    # map_partitions forces clean schema (drops author, price, etc. that may leak from merge)
    meta = pd.DataFrame(columns=out_cols)
    return merged.map_partitions(_join_partition_keep_columns, out_cols=out_cols, meta=meta)


def load_from_local(
    path: str,
    meta_path: str | None = None,
    limit: int | None = None,
    use_gpu: bool = False,
    blocksize: str = "64MB",
) -> Union[dd.DataFrame, "dask_cudf.DataFrame"]:
    """
    Load from local JSONL/JSONL.GZ.
    - If meta_path is None: return only core columns (reviewerID, asin, rating, reviewText, timestamp).
    - If meta_path is set: load both, join on parent_asin, return core + product_title, main_category.
    """
    if use_gpu and not HAS_CUDF:
        use_gpu = False

    if meta_path is None:
        # Single-file path (reviews only)
        ddf = _load_review(path, blocksize)
        ddf = ddf[CORE_COLUMNS]
    else:
        ddf_review = _load_review(path, blocksize)
        ddf_meta = _load_meta(meta_path, blocksize)
        ddf = _join_review_meta(ddf_review, ddf_meta)

    if limit is not None:
        ddf = ddf.head(limit, npartitions=-1)

    if use_gpu and HAS_CUDF:
        def _to_cudf(part: pd.DataFrame) -> "cudf.DataFrame":
            return cudf.from_pandas(part)
        meta = cudf.DataFrame(columns=ddf.columns)
        return ddf.map_partitions(_to_cudf, meta=meta)

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
