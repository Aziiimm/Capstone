"""
Optimized Load Script for GPU-Accelerated Processing.
Fixed to prevent KeyError: 'reviewerID' during filtering.
"""
from __future__ import annotations
import os
import json
from typing import Union
import pandas as pd
import dask.dataframe as dd
import dask.bag as db

try:
    import dask_cudf
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

# Schema definitions
CORE_COLUMNS = ["reviewerID", "asin", "rating", "reviewText", "timestamp"]
JOIN_KEY = "parent_asin"

# Mapping from Amazon 2023 raw schema to standardized senior project schema
REVIEW_COLUMN_MAP = {
    "user_id": "reviewerID",
    "text": "reviewText",
    "parent_asin": "parent_asin",
    "asin": "asin",
    "rating": "rating",
    "timestamp": "timestamp",
}

def _clean_and_rename(df):
    """Unified helper to handle column selection and renaming for both Test and Production."""
    # 1. Keep source columns we need (before renaming)
    src_cols = [c for c in df.columns if c in REVIEW_COLUMN_MAP.keys()]
    df = df[src_cols]
    
    # 2. Standardize names (e.g., user_id -> reviewerID)
    df = df.rename(columns=REVIEW_COLUMN_MAP)
    
    # 3. Ensure JOIN_KEY exists (required for metadata merge)
    if JOIN_KEY not in df.columns and "asin" in df.columns:
        df = df.assign(**{JOIN_KEY: df["asin"]})
    
    # 4. Final selection: only core columns for the recommender
    final_cols = [c for c in CORE_COLUMNS if c in df.columns] + [JOIN_KEY]
    return df[list(set(final_cols))]

def _load_review(path: str, blocksize: str, use_gpu: bool) -> Union[dd.DataFrame, dask_cudf.DataFrame]:
    if use_gpu and HAS_CUDF:
        ddf = dask_cudf.read_json(path, lines=True, blocksize=blocksize)
    else:
        ddf = dd.read_json(path, lines=True, blocksize=blocksize)
    return _clean_and_rename(ddf)

def _load_meta(path: str, blocksize: str, use_gpu: bool, limit: int | None = None) -> Union[dd.DataFrame, dask_cudf.DataFrame]:
    def extract_fields(line):
        try:
            data = json.loads(line)
            return {
                JOIN_KEY: str(data.get("parent_asin", "")),
                "product_title": str(data.get("title", "")),
                "main_category": str(data.get("main_category", ""))
            }
        except: return None

    if limit is not None:
        records = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= limit * 3: break
                extracted = extract_fields(line)
                if extracted: records.append(extracted)
        pdf = pd.DataFrame(records)
        ddf = dd.from_pandas(pdf, npartitions=1)
    else:
        bag = db.read_text(path, blocksize=blocksize)
        meta_dict = {JOIN_KEY: "string", "product_title": "string", "main_category": "string"}
        ddf = bag.map(extract_fields).filter(lambda x: x is not None).to_dataframe(meta=meta_dict)

    ddf = ddf[ddf[JOIN_KEY] != ""].drop_duplicates(subset=[JOIN_KEY])

    if use_gpu and HAS_CUDF:
        import cudf
        meta_df = cudf.DataFrame({k: pd.Series(dtype='str') for k in [JOIN_KEY, "product_title", "main_category"]})
        return ddf.map_partitions(lambda p: cudf.from_pandas(p), meta=meta_df)
    return ddf

def load_from_local(path: str, 
                    meta_path: str | None = None, 
                    limit: int | None = None, 
                    use_gpu: bool = True, 
                    blocksize: str = "64MB") -> Union[dd.DataFrame, dask_cudf.DataFrame]:
    
    effective_gpu = use_gpu and HAS_CUDF
    
    if limit is not None:
        records = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= limit: break
                try: records.append(json.loads(line))
                except: continue
        
        pdf = pd.DataFrame(records)
        pdf = _clean_and_rename(pdf) # Use standardized helper

        if effective_gpu:
            import cudf
            ddf = dask_cudf.from_cudf(cudf.from_pandas(pdf), npartitions=1)
        else:
            ddf = dd.from_pandas(pdf, npartitions=1)
    else:
        ddf = _load_review(path, blocksize, effective_gpu)

    if meta_path is not None:
        ddf_meta = _load_meta(meta_path, "32MB", effective_gpu, limit=limit)
        
        # Ensure consistent types for joining strings
        ddf = ddf.assign(**{JOIN_KEY: ddf[JOIN_KEY].astype(str)})
        ddf_meta = ddf_meta.assign(**{JOIN_KEY: ddf_meta[JOIN_KEY].astype(str)})

        if limit is not None:
            ddf = ddf.merge(ddf_meta, on=JOIN_KEY, how="left")
        else:
            ddf = ddf.merge(ddf_meta.compute(), on=JOIN_KEY, how="left")

    return ddf

def load(source: str, meta: str | None = None, **kwargs) -> Union[dd.DataFrame, dask_cudf.DataFrame]:
    return load_from_local(source, meta_path=meta, **kwargs)