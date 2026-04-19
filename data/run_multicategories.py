"""
Multi-category GPU orchestrator: run load -> filter -> to_parquet sequentially 
for multiple Amazon Review categories using RAPIDS (cuDF/Dask-cuDF).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import gc
from datetime import datetime
from typing import Any, Dict, List, Optional

# Standard Dask/RAPIDS imports
import dask
import rmm
import cupy as cp
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from load import load
from filter import filter_counts
from to_parquet import to_parquet

LOGGER_NAME = "gpu_multi_orchestrator"
DEFAULT_CONFIG_PATH = "categories.json"
DATASET_DIR = "dataset"
OUTPUT_DIR_DEFAULT = "output"
LOGS_DIR = "logs"

def setup_logger(run_id: str, logs_dir: str = LOGS_DIR) -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    if logger.handlers: return logger

    log_filename = os.path.join(logs_dir, f"gpu_multi_run_{run_id}.log")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    
    fh = logging.FileHandler(log_filename); fh.setFormatter(formatter)
    sh = logging.StreamHandler(); sh.setFormatter(formatter)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    with open(os.path.abspath(config_path), "r") as f:
        return json.load(f)

def cleanup_vram(logger: logging.Logger):
    """Forcefully clear TITAN RTX memory between categories."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    logger.info("VRAM Cleaned. Ready for next category.")

def run_category(
    category: Dict[str, Any],
    *,
    output_dir: str,
    use_gpu: bool,
    limit: Optional[int],
    blocksize: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    name = category.get("name")
    review_file = category.get("review_file")
    meta_file = category.get("meta_file")

    logger.info(f"=== Starting GPU Category: {name} ===")
    
    review_path = os.path.join(DATASET_DIR, review_file)
    meta_path = os.path.join(DATASET_DIR, meta_file)
    parquet_path = os.path.join(output_dir, f"dev_{name.replace(' ', '_')}.parquet")
    
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    ddf = load(source=review_path, meta=meta_path, limit=limit, use_gpu=use_gpu, blocksize=blocksize)
    timings["load_s"] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    ddf = filter_counts(ddf)
    timings["filter_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    to_parquet(ddf, parquet_path)
    timings["to_parquet_s"] = time.perf_counter() - t0
    
    total_s = sum(timings.values())
    logger.info(f"Category {name} DONE in {total_s:.2f}s")

    del ddf
    if use_gpu: cleanup_vram(logger)

    return {"category": name, "status": "success", "timings_s": timings}

def main() -> None:
    parser = argparse.ArgumentParser(description="GPU Multi-Category Orchestrator")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--gpu", action="store_true", help="Enable TITAN RTX acceleration")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(run_id)

    if args.gpu:
        logger.info("Initializing TITAN RTX with RMM Pool (20GB)...")
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=int(20e9), 
            managed_memory=True
        )
        cluster = LocalCUDACluster(rmm_pool_size="20GB")
        client = Client(cluster)
        logger.info(f"Dask-CUDA Client Active: {client}")

    config = load_config(args.config)
    categories = config.get("categories", [])
    summary = []

    for cat in categories:
        try:
            result = run_category(
                cat, output_dir=OUTPUT_DIR_DEFAULT, use_gpu=args.gpu,
                limit=args.limit, blocksize="256MB", logger=logger
            )
            summary.append(result)
        except Exception as e:
            logger.error(f"Category {cat.get('name')} FAILED: {e}")
            if not args.gpu: break # Abort if not on GPU to prevent cascading CPU hangs

    with open(os.path.join(OUTPUT_DIR_DEFAULT, "multi_run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Multi-category run complete.")

if __name__ == "__main__":
    main()