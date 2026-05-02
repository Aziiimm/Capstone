"""Smoke tests for matrix build + evaluation helpers (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
sys.path.insert(0, str(MODELS))

from build_matrix import build_matrix_from_interaction_frame  # noqa: E402
from evaluate_recommender import temporal_holdout  # noqa: E402


def test_build_matrix_from_frame():
    df = pd.DataFrame(
        {
            "reviewerID": ["a", "a", "b", "b"],
            "asin": ["i1", "i2", "i1", "i3"],
            "rating": [4.0, 5.0, 3.0, 4.0],
        }
    )
    art = build_matrix_from_interaction_frame(df, titles_by_asin=None)
    assert art.n_users == 2
    assert art.n_items == 3
    assert art.sparse_csr.shape == (3, 2)


def test_temporal_holdout_requires_two_per_user():
    df = pd.DataFrame(
        {
            "reviewerID": ["a", "a", "b"],
            "asin": ["x", "y", "z"],
            "rating": [5.0, 4.0, 3.0],
            "timestamp": [1, 2, 1],
        }
    )
    train, test = temporal_holdout(df.sort_values(["reviewerID", "timestamp"]), 1)
    assert len(test) == 1
    assert test.iloc[0]["reviewerID"] == "a"
    assert len(train) == 2
