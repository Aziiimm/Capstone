from __future__ import annotations

"""
Week 4 (Sentiment/CPU): lightweight sentiment + hybrid score utilities.

This module intentionally uses a small lexicon-based scorer so it runs with the
current project dependencies (pandas/numpy only) and can be used as a baseline
before switching to heavier NLP models.
"""

import re
from typing import Iterable

import numpy as np
import pandas as pd


# Small starter lexicon for baseline scoring.
# You can grow this over time or swap with VADER/transformers later.
POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "love",
    "loved",
    "perfect",
    "best",
    "nice",
    "fast",
    "durable",
    "recommend",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "poor",
    "hate",
    "hated",
    "worst",
    "broken",
    "slow",
    "waste",
    "disappointed",
    "refund",
    "return",
}


def clean_text(text: str) -> str:
    """Lowercase and keep alphanumerics/spaces only."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return clean_text(text).split()


def sentiment_score(text: str) -> float:
    """
    Lexicon sentiment score in [-1, 1].

    score = (pos_count - neg_count) / (pos_count + neg_count)
    If no sentiment words are found, returns 0.0.
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    denom = pos + neg
    if denom == 0:
        return 0.0
    return float((pos - neg) / denom)


def normalize_sentiment(score: float) -> float:
    """Map sentiment from [-1, 1] to [0, 1]."""
    clipped = max(-1.0, min(1.0, float(score)))
    return (clipped + 1.0) / 2.0


def add_sentiment_columns(
    df: pd.DataFrame,
    text_col: str = "reviewText",
) -> pd.DataFrame:
    """Add `sentiment_raw` and `sentiment_norm` columns to a copy of df."""
    out = df.copy()
    if text_col not in out.columns:
        raise ValueError(f"Missing text column '{text_col}'.")
    out["sentiment_raw"] = out[text_col].fillna("").map(sentiment_score)
    out["sentiment_norm"] = out["sentiment_raw"].map(normalize_sentiment)
    return out


def add_hybrid_score(
    df: pd.DataFrame,
    rating_col: str = "rating",
    text_col: str = "reviewText",
    rating_weight: float = 0.7,
    sentiment_weight: float = 0.3,
) -> pd.DataFrame:
    """
    Add sentiment columns and `hybrid_score`:

    hybrid_score = rating_weight * rating + sentiment_weight * sentiment_norm
    """
    if not np.isclose(rating_weight + sentiment_weight, 1.0):
        raise ValueError("rating_weight + sentiment_weight must equal 1.0")
    if rating_col not in df.columns:
        raise ValueError(f"Missing rating column '{rating_col}'.")

    out = add_sentiment_columns(df, text_col=text_col)
    out["hybrid_score"] = (
        rating_weight * out[rating_col].astype(float)
        + sentiment_weight * out["sentiment_norm"].astype(float)
    )
    return out


def cold_start_profile_from_texts(texts: Iterable[str]) -> dict[str, float]:
    """
    Build a simple cold-start profile from initial user text only.

    Returns a dict with:
      - sentiment_raw_mean in [-1, 1]
      - sentiment_norm_mean in [0, 1]
    """
    vals = [sentiment_score(t) for t in texts]
    if not vals:
        raw = 0.0
    else:
        raw = float(np.mean(vals))
    return {
        "sentiment_raw_mean": raw,
        "sentiment_norm_mean": normalize_sentiment(raw),
    }

