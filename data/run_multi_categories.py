"""
Multi-category orchestrator: run load -> filter -> to_parquet sequentially for multiple
Amazon Review categories, with logging, per-category timings, and optional summary.
Input files in the dataset directory are deleted only after ALL categories complete successfully.
"""
from __future__ import annotations

import argparse
import dask
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from data.load import load
from data.filter import filter_counts
from data.to_parquet import to_parquet


LOGGER_NAME = "multi_categories_safe"
DEFAULT_CONFIG_PATH = "categories.json"
DATASET_DIR = "dataset"
OUTPUT_DIR_DEFAULT = "output"
LOGS_DIR = "logs"


def setup_logger(run_id: str, logs_dir: str = LOGS_DIR) -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    log_filename = os.path.join(logs_dir, f"multi_run_safe_{run_id}.log")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logging initialized. Log file: %s", log_filename)
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    resolved = os.path.abspath(config_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with open(resolved, "r") as f:
        data = json.load(f)
    if "categories" not in data or not isinstance(data["categories"], list):
        raise ValueError(
            f"Config {resolved} must contain a 'categories' list. "
            "See README for expected structure."
        )
    return data


def _ensure_under_dataset(path: str) -> bool:
    """Return True if the given path is safely under the dataset directory."""
    dataset_abs = os.path.abspath(DATASET_DIR)
    target_abs = os.path.abspath(path)
    return os.path.commonpath([dataset_abs, target_abs]) == dataset_abs


def _delete_input_file(path: str, logger: logging.Logger) -> None:
    if not os.path.exists(path):
        logger.info("Input file already missing (nothing to delete): %s", path)
        return
    if not _ensure_under_dataset(path):
        logger.warning(
            "Refusing to delete file outside dataset directory: %s", path
        )
        return
    try:
        os.remove(path)
        logger.info("Deleted input file: %s", path)
    except OSError as exc:
        logger.error("Failed to delete input file %s: %s", path, exc)


def cleanup_dataset(
    categories: List[Dict[str, Any]],
    logger: logging.Logger,
) -> None:
    """
    Delete all review and meta input files for every category.
    Only called after all categories have completed successfully.
    """
    logger.info("All categories completed -- starting dataset cleanup.")
    for category in categories:
        name = category.get("name", "<unknown>")
        review_file = category.get("review_file")
        meta_file = category.get("meta_file")
        if review_file:
            _delete_input_file(os.path.join(DATASET_DIR, review_file), logger)
        if meta_file:
            _delete_input_file(os.path.join(DATASET_DIR, meta_file), logger)
        logger.info("Category %s: input files cleaned up.", name)
    logger.info("Dataset cleanup complete.")


def run_category(
    category: Dict[str, Any],
    *,
    output_dir: str,
    use_gpu: bool,
    limit: Optional[int],
    blocksize: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Run the pipeline for a single category.
    Returns a dict with status, timings, and paths.
    Input files are never deleted here -- cleanup happens after all categories succeed.
    """
    name = category.get("name")
    review_file = category.get("review_file")
    meta_file = category.get("meta_file")

    if not name or not review_file or not meta_file:
        raise ValueError(
            f"Category entry must have 'name', 'review_file', and 'meta_file'. Got: {category}"
        )

    logger.info("=== Starting category: %s ===", name)

    review_path = os.path.join(DATASET_DIR, review_file)
    meta_path = os.path.join(DATASET_DIR, meta_file)

    if not os.path.isfile(review_path):
        raise FileNotFoundError(f"Review file not found for {name}: {review_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Meta file not found for {name}: {meta_path}")

    os.makedirs(output_dir, exist_ok=True)

    safe_name = str(name).replace(" ", "_")
    parquet_path = os.path.join(output_dir, f"dev_{safe_name}.parquet")
    timings_path = os.path.join(output_dir, f"timings_{safe_name}.json")

    timings: Dict[str, float] = {}
    result: Dict[str, Any] = {
        "category": name,
        "review_path": review_path,
        "meta_path": meta_path,
        "parquet_path": parquet_path,
        "timings_path": timings_path,
        "backend": "gpu" if use_gpu else "cpu",
        "status": "started",
    }

    logger.info(
        "Category %s: review=%s meta=%s output_parquet=%s",
        name,
        review_path,
        meta_path,
        parquet_path,
    )

    import time

    t0 = time.perf_counter()
    ddf = load(
        source=review_path,
        meta=meta_path,
        limit=limit,
        use_gpu=use_gpu,
        blocksize=blocksize,
    )
    load_s = time.perf_counter() - t0
    timings["load_s"] = load_s
    logger.info("Category %s: load completed in %.2fs", name, load_s)

    t0 = time.perf_counter()
    ddf = filter_counts(ddf)
    filter_s = time.perf_counter() - t0
    timings["filter_s"] = filter_s
    logger.info("Category %s: filter completed in %.2fs", name, filter_s)

    t0 = time.perf_counter()
    to_parquet(ddf, parquet_path)
    to_parquet_s = time.perf_counter() - t0
    timings["to_parquet_s"] = to_parquet_s
    logger.info("Category %s: to_parquet completed in %.2fs", name, to_parquet_s)

    total_s = load_s + filter_s + to_parquet_s
    timings["total_s"] = total_s
    logger.info("Category %s: DONE in %.2fs", name, total_s)

    with open(timings_path, "w") as f:
        json.dump(
            {
                "backend": result["backend"],
                "category": name,
                "source": review_path,
                "meta": meta_path,
                "output": parquet_path,
                "limit": limit,
                "timings_s": timings,
            },
            f,
            indent=2,
        )
    logger.info("Category %s: timings written to %s", name, timings_path)

    result["timings_s"] = timings
    result["status"] = "success"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the data pipeline for multiple categories sequentially, using a JSON "
            "config file. Input files are deleted only after ALL categories succeed."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config listing categories (default: categories.json).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU (Dask-cuDF) if available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: limit rows per category (for testing). Omit for full dataset.",
    )
    parser.add_argument(
        "--blocksize",
        default="256MB",
        help="Dask read block size (default 256MB).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_DEFAULT,
        help="Base directory for Parquet and timings outputs (default: output/).",
    )
    parser.add_argument(
        "--summary-timings",
        default=None,
        help=(
            "Optional path to a JSON file that will collect a summary of all categories "
            "(default: output/timings_multi_summary.json)."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "If set, continue with remaining categories even if one category fails. "
            "By default, the multi-run aborts on the first failure."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of threads for the Dask threaded scheduler (e.g. --workers 4). "
            "If omitted, Dask uses its default single-threaded scheduler. "
            "Run 'nproc' to see available cores."
        ),
    )
    parser.add_argument(
        "--no-delete-inputs",
        action="store_true",
        help=(
            "If set, do NOT delete dataset input files after all categories succeed. "
            "Useful for debugging or rerunning."
        ),
    )

    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(run_id)

    # Configure Dask threaded scheduler if workers specified
    if args.workers is not None:
        dask.config.set(scheduler="threads", num_workers=args.workers)
        logger.info("Using threaded scheduler with %d threads.", args.workers)
    else:
        logger.info("No --workers specified, using Dask default scheduler.")

    logger.info("Multi-category run started with config=%s", args.config)
    config = load_config(args.config)
    categories: List[Dict[str, Any]] = config.get("categories", [])

    if not categories:
        logger.error("No categories found in config %s", args.config)
        raise SystemExit(1)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    any_failure = False

    for cat in categories:
        name = cat.get("name", "<unknown>")
        try:
            result = run_category(
                cat,
                output_dir=output_dir,
                use_gpu=args.gpu,
                limit=args.limit,
                blocksize=args.blocksize,
                logger=logger,
            )
            summary.append(result)
        except Exception as exc:
            any_failure = True
            logger.exception("Category %s FAILED: %s", name, exc)
            summary.append(
                {
                    "category": name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            if not args.continue_on_error:
                logger.error(
                    "Aborting multi-run due to failure and --continue-on-error not set."
                )
                break

    # Only delete inputs if ALL categories succeeded and --no-delete-inputs not set
    if not any_failure and not args.no_delete_inputs:
        cleanup_dataset(categories, logger)
    elif any_failure:
        logger.warning(
            "Skipping dataset cleanup due to one or more category failures. "
            "Input files have been preserved."
        )
    elif args.no_delete_inputs:
        logger.info("--no-delete-inputs set, skipping dataset cleanup.")

    summary_path: Optional[str]
    if args.summary_timings is not None:
        summary_path = args.summary_timings
    else:
        summary_path = os.path.join(output_dir, "timings_multi_summary.json")

    try:
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "backend": "gpu" if args.gpu else "cpu",
                    "workers": args.workers,
                    "config": os.path.abspath(args.config),
                    "output_dir": os.path.abspath(output_dir),
                    "continue_on_error": args.continue_on_error,
                    "delete_inputs": not args.no_delete_inputs,
                    "categories": summary,
                },
                f,
                indent=2,
            )
        logger.info("Multi-run summary written to %s", summary_path)
    except OSError as exc:
        logger.error("Failed to write summary timings file %s: %s", summary_path, exc)

    if any_failure:
        logger.error("Multi-category run completed with failures.")
        raise SystemExit(1)

    logger.info("Multi-category run completed successfully.")


if __name__ == "__main__":
    main()