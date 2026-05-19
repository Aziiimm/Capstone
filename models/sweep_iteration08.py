"""
Run iteration-08-only configs (faster than full sweep).

  cd models && py -3 sweep_iteration08.py
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
_EVAL = _MODEL_DIR / "evaluate_ranking.py"
_HIT = re.compile(r"Hit Rate@(\d+):\s*([\d.]+)")
_MRR = re.compile(r"MRR@(\d+):\s*([\d.]+)")

_BASE = (
    "--split", "temporal", "--bm25-weighting", "--rating-weighted-merge", "--neighbors", "10",
)

CONFIGS: list[tuple[str, list[str]]] = [
    ("iter07_best_ensemble_w35", ["--model", "ensemble", "--ensemble-weight", "0.35", *_BASE]),
    ("stack_default", ["--model", "stack", *_BASE]),
    ("stack_w20_50_30", ["--model", "stack", "--stack-weights", "0.2,0.5,0.3", *_BASE]),
    ("stack_w15_45_40", ["--model", "stack", "--stack-weights", "0.15,0.45,0.40", *_BASE]),
    ("ensemble_w35_content15", [
        "--model", "ensemble", "--ensemble-weight", "0.35", "--content-hybrid",
        "--content-weight", "0.15", *_BASE,
    ]),
    ("ensemble_w35_content10", [
        "--model", "ensemble", "--ensemble-weight", "0.35", "--content-hybrid",
        "--content-weight", "0.10", *_BASE,
    ]),
    ("ensemble_w35_content20", [
        "--model", "ensemble", "--ensemble-weight", "0.35", "--content-hybrid",
        "--content-weight", "0.20", *_BASE,
    ]),
    ("stack_content15", ["--model", "stack", "--content-hybrid", "--content-weight", "0.15", *_BASE]),
    ("nmf64_bm25_temporal", ["--model", "nmf", "--nmf-components", "64", *_BASE]),
    ("svd96_bm25_temporal", ["--model", "svd", "--svd-components", "96", *_BASE]),
    ("als_bm25_temporal_alpha40", [
        "--model", "als", "--als-confidence-alpha", "40", *_BASE,
    ]),
    ("als_bm25_temporal_alpha80", [
        "--model", "als", "--als-confidence-alpha", "80", *_BASE,
    ]),
    ("knn_euclidean_bm25_rw", ["--knn-metric", "euclidean", *_BASE]),
    ("bm25_k1_50", ["--bm25-k1", "50", "--ensemble-weight", "0.35", "--model", "ensemble", *_BASE]),
    ("bm25_k1_200", ["--bm25-k1", "200", "--ensemble-weight", "0.35", "--model", "ensemble", *_BASE]),
]

DATASETS = [
    "../output/tools_gpu_50k.parquet",
    "../output/industrial_dev_50k.parquet",
    "../output/sample_clean.parquet",
    "../output/dev_tools.parquet",
]


def main() -> None:
    out_path = (_MODEL_DIR / "../output/sweep_iteration08.csv").resolve()
    rows = []
    for ds in DATASETS:
        p = (_MODEL_DIR / ds).resolve()
        if not p.exists():
            print(f"SKIP {p}")
            continue
        print(f"\n=== {p.name} ===")
        for name, extra in CONFIGS:
            cmd = [sys.executable, str(_EVAL), "--parquet", str(p), "--skip-baselines", *extra]
            proc = subprocess.run(cmd, cwd=str(_MODEL_DIR), capture_output=True, text=True, timeout=600)
            out = proc.stdout + proc.stderr
            hm, mm = _HIT.search(out), _MRR.search(out)
            hit = hm.group(2) if hm else ""
            mrr = mm.group(2) if mm else ""
            err = "" if proc.returncode == 0 else out.strip()[-300:]
            rows.append({"dataset": p.name, "config": name, "hit_rate": hit, "mrr": mrr, "error": err})
            print(f"  {name}: hit={hit or 'ERR'} mrr={mrr or ''}")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "config", "hit_rate", "mrr", "error"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path}")

    by_ds: dict[str, list] = {}
    for r in rows:
        if r["hit_rate"]:
            by_ds.setdefault(r["dataset"], []).append(r)
    print("\n--- Best per dataset ---")
    for ds, rs in sorted(by_ds.items()):
        b = max(rs, key=lambda x: float(x["hit_rate"]))
        print(f"  {ds}: {b['config']} hit={b['hit_rate']} mrr={b['mrr']}")


if __name__ == "__main__":
    main()
