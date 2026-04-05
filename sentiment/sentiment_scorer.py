"""
Read a cleaned Parquet file and compute VADER sentiment for every review to build a hybrid score:
hybrid_score = 0.7 * rating_norm + 0.3 * sentiment_norm
and writes the enriched Parquet back to disk.

"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Text preprocessing
def clean_text(series: pd.Series) -> pd.Series:
    """Lowercase and strip punctuation from review text."""
    log.info("Cleaning text …")
    t0 = time.perf_counter()
    cleaned = (
        series.fillna("")
              .str.lower()
              .str.replace(r"[^\w\s]", " ", regex=True)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
    )
    log.info("  text clean: %.2fs", time.perf_counter() - t0)
    return cleaned

# VADER sentiment scoring: return compound VADER score [-1,1] for each text.
def score_vader(texts: pd.Series) -> pd.Series:
    log.info("Running VADER sentiment on %d rows …", len(texts))
    t0 = time.perf_counter()
    analyzer = SentimentIntensityAnalyzer()

    BATCH = 50_000
    scores: list[float] = []
    for start in range(0, len(texts), BATCH):
        batch = texts.iloc[start : start + BATCH]
        scores.extend(analyzer.polarity_scores(t)["compound"] for t in batch)
        log.info("  scored %d / %d rows", min(start + BATCH, len(texts)), len(texts))

    elapsed = time.perf_counter() - t0
    log.info("  VADER done: %.2fs (%.0f rows/s)", elapsed, len(texts) / elapsed)
    return pd.Series(scores, index=texts.index, name="sentiment_compound")

# Normalise rating (1-5 → 0-1) and sentiment (−1..+1 → 0-1)
def build_hybrid_score(
    df: pd.DataFrame,
    rating_col: str = "rating",
    sentiment_col: str = "sentiment_compound",
    rating_weight: float = 0.7,
    sentiment_weight: float = 0.3,
) -> pd.Series:
    rating_norm = (pd.to_numeric(df[rating_col], errors="coerce") - 1.0) / 4.0
    sentiment_norm = (pd.to_numeric(df[sentiment_col], errors="coerce") + 1.0) / 2.0
    hybrid = rating_weight * rating_norm + sentiment_weight * sentiment_norm
    return hybrid.rename("hybrid_score")

# Cold-start recommendations: for new users, rank products by their average hybrid score
# Only products with >= min_reviews are considered.
# Returns a DataFrame with columns: asin, product_title, main_category, avg_hybrid_score, avg_sentiment, avg_rating, review_count
def cold_start_recommendations(
    df: pd.DataFrame,
    top_k: int = 10,
    min_reviews: int = 20,
) -> pd.DataFrame:
    agg = (
        df.groupby("asin")
          .agg(
              avg_hybrid_score=("hybrid_score", "mean"),
              avg_sentiment=("sentiment_compound", "mean"),
              avg_rating=("rating", "mean"),
              review_count=("rating", "count"),
          )
          .reset_index()
    )
    str_cols = [c for c in ["product_title", "main_category"] if c in df.columns]
    if str_cols:
        str_agg = df.groupby("asin")[str_cols].first().reset_index()
        agg = agg.merge(str_agg, on="asin", how="left")
    popular = agg[agg["review_count"] >= min_reviews]
    top = popular.nlargest(top_k, "avg_hybrid_score").reset_index(drop=True)
    return top

# Pipeline to score sentiment and build hybrid score
def run(input_path: str, output_path: str) -> dict:
    timings: dict[str, float] = {}

    # Load the cleaned Parquet file
    log.info("Loading %s …", input_path)
    t0 = time.perf_counter()
    df = pd.read_parquet(input_path)
    timings["load_s"] = round(time.perf_counter() - t0, 3)
    log.info("  loaded %d rows, %d cols in %.2fs", *df.shape, timings["load_s"])

    required = {"reviewerID", "asin", "rating", "reviewText"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input Parquet missing columns: {missing}")

    # Ensure rating is numeric (parquet may load it as large_string via Arrow)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Text preprocessing
    t0 = time.perf_counter()
    df["clean_text"] = clean_text(df["reviewText"])
    timings["text_clean_s"] = round(time.perf_counter() - t0, 3)

    # Sentiment scoring
    t0 = time.perf_counter()
    df["sentiment_compound"] = score_vader(df["clean_text"])
    timings["sentiment_s"] = round(time.perf_counter() - t0, 3)

    # Hybrid score fusion
    t0 = time.perf_counter()
    df["hybrid_score"] = build_hybrid_score(df)
    timings["hybrid_score_s"] = round(time.perf_counter() - t0, 3)

    # Cold-start recommendations
    if "product_title" in df.columns and "main_category" in df.columns:
        cold_start = cold_start_recommendations(df, top_k=10)
        log.info("Cold-start top-10 by hybrid score:\n%s", cold_start.to_string(index=False))

    # Write the enriched Parquet file
    log.info("Writing enriched Parquet → %s", output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    df.drop(columns=["clean_text"]).to_parquet(output_path, index=False)
    timings["write_s"] = round(time.perf_counter() - t0, 3)

    timings["total_s"] = round(sum(timings.values()), 3)
    log.info("Done. Timings: %s", timings)
    return timings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 4 – CPU sentiment scorer")
    p.add_argument("--input",  required=True, help="Path to cleaned .parquet")
    p.add_argument("--output", required=True, help="Path for enriched .parquet")
    p.add_argument("--timings-file", default=None, help="Optional JSON file for timings")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    timings = run(args.input, args.output)
    if args.timings_file:
        Path(args.timings_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.timings_file, "w") as f:
            json.dump({"backend": "cpu", "stage": "sentiment", **timings}, f, indent=2)
        log.info("Timings saved → %s", args.timings_file)


if __name__ == "__main__":
    main()
