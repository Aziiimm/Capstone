"""
PRIMARY MODEL  : SVD (Singular Value Decomposition) via scikit-surprise
COLD-START     : Hybrid popularity ranking (sentiment-aware)

The trained SVD model is handed to AmazonRecommender (models/recommender.py),
which handles all inference.

1. Load enriched Parquet (with hybrid_score from sentiment_scorer.py)
2. Train SVD on hybrid_score
3. Save model and metrics JSON

python -m models.cpu_baseline \
    --input     output/dev_Electronics_sentiment.parquet \
    --category  Electronics \
    --output-dir output \
    --model-dir  models/saved
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from models.recommender import AmazonRecommender

try:
    from surprise import Dataset, Reader, SVD, accuracy
    from surprise.model_selection import cross_validate, train_test_split
    HAS_SURPRISE = True
except ImportError:
    HAS_SURPRISE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Helper functions
def load_data(path: str) -> pd.DataFrame:
    log.info("Loading %s …", path)
    df = pd.read_parquet(path)
    required = {"reviewerID", "asin", "hybrid_score"}
    if missing := required - set(df.columns):
        raise ValueError(
            f"Missing columns: {missing}. "
            "Run sentiment_scorer.py first to generate hybrid_score."
        )
    log.info("  %d rows loaded", len(df))
    return df


def build_surprise_dataset(df: pd.DataFrame) -> "Dataset":
    """Convert DataFrame to a Surprise Dataset using hybrid_score as the rating."""
    reader = Reader(rating_scale=(0.0, 1.0))
    return Dataset.load_from_df(df[["reviewerID", "asin", "hybrid_score"]], reader)

# Evaluation metrics
NDCG_K = 10
PRECISION_K = 10

def _dcg(relevances: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def ndcg_at_k(predictions, k: int = NDCG_K) -> float:
    user_preds: dict = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_preds[uid].append((est, true_r))
    ndcgs = []
    for preds in user_preds.values():
        top = sorted(preds, key=lambda x: x[0], reverse=True)[:k]
        ideal = sorted(preds, key=lambda x: x[1], reverse=True)[:k]
        dcg = _dcg([r for _, r in top])
        idcg = _dcg([r for _, r in ideal])
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def precision_at_k(predictions, k: int = PRECISION_K, threshold: float = 0.6) -> float:
    user_preds: dict = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_preds[uid].append((est, true_r))
    precisions = []
    for preds in user_preds.values():
        top_k = sorted(preds, key=lambda x: x[0], reverse=True)[:k]
        precisions.append(sum(1 for _, r in top_k if r >= threshold) / k)
    return float(np.mean(precisions)) if precisions else 0.0


def _evaluate(algo, testset) -> dict:
    t0 = time.perf_counter()
    preds = algo.test(testset)
    infer_time = time.perf_counter() - t0
    return {
        "rmse": round(accuracy.rmse(preds, verbose=False), 4),
        "mae": round(accuracy.mae(preds, verbose=False), 4),
        f"ndcg_at_{NDCG_K}": round(ndcg_at_k(preds), 4),
        f"precision_at_{PRECISION_K}": round(precision_at_k(preds), 4),
        "infer_time_s": round(infer_time, 3),
    }

# SVD Primary Recommender Model: Matrix Factorisation
def train_svd(trainset, testset) -> tuple[object, dict]:
    log.info("Training SVD (PRIMARY recommender) …")
    t0 = time.perf_counter()
    algo = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
    )
    algo.fit(trainset)
    train_time = time.perf_counter() - t0
    log.info("  SVD trained in %.2fs", train_time)

    metrics = {
        "model": "SVD",
        "role": "primary",
        "train_time_s": round(train_time, 3),
        **_evaluate(algo, testset),
    }
    log.info(
        "  SVD → RMSE=%.4f  NDCG@10=%.4f  Prec@10=%.4f",
        metrics["rmse"], metrics[f"ndcg_at_{NDCG_K}"], metrics[f"precision_at_{PRECISION_K}"],
    )
    return algo, metrics

def run(
    input_path: str,
    category: str,
    output_dir: str,
    model_dir: str,
) -> dict:
    if not HAS_SURPRISE:
        raise ImportError("Run: pip install scikit-surprise")

    df = load_data(input_path)
    data = build_surprise_dataset(df)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    svd_algo, svd_metrics = train_svd(trainset, testset)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    svd_path = Path(model_dir) / f"{category}_svd.pkl"
    with open(svd_path, "wb") as f:
        pickle.dump(svd_algo, f)
    log.info("Pickle saved → %s", svd_path)

    recommender = AmazonRecommender(svd_algo, df, category)
    recommender.save(str(Path(model_dir) / category))

    all_metrics = {
        "backend": "cpu",
        "category": category,
        "primary_model": "SVD",
        "n_rows": len(df),
        "n_users": df["reviewerID"].nunique(),
        "n_items": df["asin"].nunique(),
        "models": [svd_metrics],
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(output_dir) / f"metrics_cpu_{category}.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)

    return all_metrics

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SVD recommender")
    p.add_argument("--input", default=None,
                   help="Path to sentiment parquet (required unless --infer-only)")
    p.add_argument("--category", required=True)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--model-dir", default="models/saved")
    p.add_argument("--demo-user", default=None,
                   help="Print top-10 recs for this user ID")
    p.add_argument("--infer-only", action="store_true",
                   help="Load trained model and infer")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.infer_only:
        if not args.demo_user:
            raise ValueError("--infer-only requires --demo-user to be set")
        log.info("Skipping training — loading existing model for %s", args.category)
    else:
        if not args.input:
            raise ValueError("--input is required unless --infer-only is set")
        run(args.input, args.category, args.output_dir, args.model_dir)

    if args.demo_user:
        prefix = str(Path(args.model_dir) / args.category)
        rec = AmazonRecommender.load(prefix)
        recs = rec.recommend(args.demo_user, top_k=10)
        print(f"\nTop-10 recommendations for '{args.demo_user}':")
        print(recs.to_string(index=False))


if __name__ == "__main__":
    main()