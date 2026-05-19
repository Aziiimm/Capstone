"""
Hyperparameter / model sweep for evaluate_ranking.py.

Runs many configs across one or more Parquet datasets and writes a CSV summary.
Best config per dataset is printed at the end.

Example:
  cd models
  py -3 sweep_ranking.py --datasets ../output/dev_tools.parquet ../output/industrial_dev_50k.parquet
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
_EVAL_SCRIPT = _MODEL_DIR / "evaluate_ranking.py"

_HIT_RE = re.compile(r"Hit Rate@(\d+):\s*([\d.]+)")
_MRR_RE = re.compile(r"MRR@(\d+):\s*([\d.]+)")
_CASES_RE = re.compile(r"Cases evaluated:\s*(\d+)")


@dataclass(frozen=True)
class SweepConfig:
    name: str
    extra_args: tuple[str, ...] = ()


# Core model variants (neighbor count swept separately).
_BASE_CONFIGS: list[SweepConfig] = [
    SweepConfig("knn_default"),
    SweepConfig("knn_no_center", ("--no-center-users",)),
    SweepConfig("knn_implicit", ("--implicit-matrix",)),
    SweepConfig("knn_bm25", ("--bm25-weighting",)),
    SweepConfig("knn_rating_merge", ("--rating-weighted-merge",)),
    SweepConfig("knn_implicit_rw", ("--implicit-matrix", "--rating-weighted-merge")),
    SweepConfig("knn_bm25_rw", ("--bm25-weighting", "--rating-weighted-merge")),
    SweepConfig("knn_sentiment", ("--sentiment-weighting",)),
    SweepConfig("knn_sentiment_rw", ("--sentiment-weighting", "--rating-weighted-merge")),
    SweepConfig("svd32", ("--model", "svd", "--svd-components", "32")),
    SweepConfig("svd64", ("--model", "svd", "--svd-components", "64")),
    SweepConfig("svd128", ("--model", "svd", "--svd-components", "128")),
    SweepConfig("svd64_bm25", ("--model", "svd", "--svd-components", "64", "--bm25-weighting")),
    SweepConfig("svd64_rw", ("--model", "svd", "--svd-components", "64", "--rating-weighted-merge")),
    SweepConfig("svd64_bm25_rw", (
        "--model", "svd", "--svd-components", "64", "--bm25-weighting", "--rating-weighted-merge",
    )),
    # Iteration 06+
    SweepConfig("als_bm25", ("--model", "als", "--bm25-weighting")),
    SweepConfig("als_bm25_temporal", ("--model", "als", "--bm25-weighting", "--split", "temporal")),
    SweepConfig("ensemble_temporal", (
        "--model", "ensemble", "--split", "temporal",
        "--bm25-weighting", "--rating-weighted-merge", "--neighbors", "10",
    )),
    SweepConfig("ensemble_temporal_w35", (
        "--model", "ensemble", "--split", "temporal",
        "--bm25-weighting", "--rating-weighted-merge", "--neighbors", "10",
        "--ensemble-weight", "0.35",
    )),
    SweepConfig("ensemble_temporal_w30_q30", (
        "--model", "ensemble", "--split", "temporal",
        "--bm25-weighting", "--rating-weighted-merge", "--neighbors", "10",
        "--ensemble-weight", "0.3", "--query-neighbors", "30",
    )),
    SweepConfig("knn_bm25_rw_n10_temporal_pop", (
        "--split", "temporal", "--bm25-weighting", "--rating-weighted-merge",
        "--neighbors", "10", "--popularity-prior",
    )),
    SweepConfig("knn_bm25_rw_n10_temporal_recency", (
        "--split", "temporal", "--bm25-weighting", "--rating-weighted-merge",
        "--neighbors", "10", "--recency-half-life-days", "180",
    )),
    SweepConfig("knn_bm25_rw_n10_min4", (
        "--bm25-weighting", "--rating-weighted-merge", "--neighbors", "10",
        "--min-history-rating", "4",
    )),
    # Iteration 08+
    SweepConfig("stack_temporal", (
        "--model", "stack", "--split", "temporal",
        "--bm25-weighting", "--rating-weighted-merge", "--neighbors", "10",
    )),
    SweepConfig("ensemble_w35_content20_temporal", (
        "--model", "ensemble", "--ensemble-weight", "0.35", "--content-hybrid",
        "--content-weight", "0.20", "--split", "temporal",
        "--bm25-weighting", "--rating-weighted-merge", "--neighbors", "10",
    )),
    SweepConfig("svd96_bm25_temporal", (
        "--model", "svd", "--svd-components", "96", "--split", "temporal",
        "--bm25-weighting", "--rating-weighted-merge", "--neighbors", "10",
    )),
    SweepConfig("nmf64_bm25_temporal", (
        "--model", "nmf", "--nmf-components", "64", "--split", "temporal",
        "--bm25-weighting", "--rating-weighted-merge", "--neighbors", "10",
    )),
]


def _neighbor_variants(neighbors: list[int]) -> list[SweepConfig]:
    out: list[SweepConfig] = []
    for k in neighbors:
        out.append(SweepConfig(f"knn_n{k}", ("--neighbors", str(k),)))
        out.append(SweepConfig(f"knn_bm25_rw_n{k}", (
            "--bm25-weighting", "--rating-weighted-merge", "--neighbors", str(k),
        )))
    return out


def _temporal_variants(names: list[str]) -> list[SweepConfig]:
    """Re-run top configs with temporal split."""
    lookup = {c.name: c for c in _BASE_CONFIGS + _neighbor_variants([5, 10, 20])}
    out: list[SweepConfig] = []
    for name in names:
        base = lookup.get(name)
        if base is None:
            continue
        out.append(SweepConfig(
            f"{name}_temporal",
            base.extra_args + ("--split", "temporal"),
        ))
    return out


def build_configs(*, neighbors: list[int], temporal_tops: list[str]) -> list[SweepConfig]:
    seen: set[str] = set()
    configs: list[SweepConfig] = []
    for c in _BASE_CONFIGS + _neighbor_variants(neighbors):
        if c.name not in seen:
            seen.add(c.name)
            configs.append(c)
    for c in _temporal_variants(temporal_tops):
        if c.name not in seen:
            seen.add(c.name)
            configs.append(c)
    return configs


def _run_one(
    parquet: str,
    config: SweepConfig,
    *,
    top_k: int,
    seed: int,
    test_fraction: float,
    python: str,
) -> dict:
    cmd = [
        python,
        str(_EVAL_SCRIPT),
        "--parquet", parquet,
        "--top-k", str(top_k),
        "--seed", str(seed),
        "--test-fraction", str(test_fraction),
        "--skip-baselines",
        *config.extra_args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(_MODEL_DIR),
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        return {
            "dataset": parquet,
            "config": config.name,
            "error": out.strip()[-500:] or f"exit {proc.returncode}",
            "hit_rate": "",
            "mrr": "",
            "cases": "",
        }

    hit_m = _HIT_RE.search(out)
    mrr_m = _MRR_RE.search(out)
    cases_m = _CASES_RE.search(out)
    return {
        "dataset": parquet,
        "config": config.name,
        "error": "",
        "hit_rate": hit_m.group(2) if hit_m else "",
        "mrr": mrr_m.group(2) if mrr_m else "",
        "cases": cases_m.group(1) if cases_m else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep evaluate_ranking configs across datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "../output/dev_tools.parquet",
            "../output/industrial_dev_50k.parquet",
            "../output/tools_gpu_50k.parquet",
            "../output/sample_clean.parquet",
        ],
        help="Parquet paths relative to models/ or absolute",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--neighbors", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument(
        "--temporal-tops",
        nargs="*",
        default=["knn_default", "knn_bm25_rw", "svd64", "knn_bm25_rw_n10"],
        help="Config names to also run with --split temporal",
    )
    parser.add_argument(
        "--out",
        default="../output/sweep_results.csv",
        help="CSV output path (relative to models/ unless absolute)",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    configs = build_configs(neighbors=args.neighbors, temporal_tops=args.temporal_tops)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (_MODEL_DIR / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total = len(args.datasets) * len(configs)
    n_done = 0
    for parquet in args.datasets:
        p = Path(parquet)
        if not p.is_absolute():
            p = (_MODEL_DIR / parquet).resolve()
        if not p.exists() and not p.parent.exists():
            print(f"SKIP missing: {p}")
            continue
        parquet_str = str(p)
        print(f"\n=== Dataset: {p.name} ===")
        for cfg in configs:
            n_done += 1
            print(f"  [{n_done}/{total}] {cfg.name} ...", flush=True)
            row = _run_one(
                parquet_str,
                cfg,
                top_k=args.top_k,
                seed=args.seed,
                test_fraction=args.test_fraction,
                python=args.python,
            )
            rows.append(row)
            if row["error"]:
                print(f"    ERROR: {row['error'][:120]}")
            elif row["hit_rate"]:
                print(f"    Hit@{args.top_k}={row['hit_rate']}  MRR={row['mrr']}")

    fieldnames = ["dataset", "config", "hit_rate", "mrr", "cases", "error"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_path}")

    # Best per dataset by hit_rate
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("error") or not r.get("hit_rate"):
            continue
        ds = Path(r["dataset"]).name
        by_ds.setdefault(ds, []).append(r)

    print("\n--- Best config per dataset (by Hit Rate) ---")
    for ds, rs in sorted(by_ds.items()):
        best = max(rs, key=lambda x: float(x["hit_rate"]))
        print(
            f"  {ds}: {best['config']}  "
            f"Hit@{args.top_k}={best['hit_rate']}  MRR={best['mrr']}  (n={best['cases']})"
        )


if __name__ == "__main__":
    main()
