from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


_root = Path(__file__).resolve().parent.parent
_sent_py = _root / "features" / "sentiment.py"
_spec = importlib.util.spec_from_file_location("_features_sentiment", _sent_py)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

sentiment_score = _mod.sentiment_score
normalize_sentiment = _mod.normalize_sentiment
add_hybrid_score = _mod.add_hybrid_score
cold_start_profile_from_texts = _mod.cold_start_profile_from_texts


def test_sentiment_score_polarity():
    assert sentiment_score("excellent amazing good") > 0
    assert sentiment_score("terrible bad worst") < 0
    assert sentiment_score("table chair bottle") == 0.0


def test_normalize_sentiment_range():
    assert normalize_sentiment(-1.0) == 0.0
    assert normalize_sentiment(0.0) == 0.5
    assert normalize_sentiment(1.0) == 1.0


def test_add_hybrid_score_columns():
    df = pd.DataFrame(
        [
            {"reviewerID": "U1", "asin": "I1", "rating": 5.0, "reviewText": "great product"},
            {"reviewerID": "U2", "asin": "I2", "rating": 2.0, "reviewText": "bad and broken"},
        ]
    )
    out = add_hybrid_score(df, rating_weight=0.7, sentiment_weight=0.3)
    assert {"sentiment_raw", "sentiment_norm", "hybrid_score"}.issubset(out.columns)
    assert out.loc[0, "hybrid_score"] > out.loc[1, "hybrid_score"]


def test_cold_start_profile():
    p = cold_start_profile_from_texts(["good quality", "excellent choice"])
    assert p["sentiment_raw_mean"] > 0
    assert 0 <= p["sentiment_norm_mean"] <= 1

