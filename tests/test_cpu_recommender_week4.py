from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


_root = Path(__file__).resolve().parent.parent
_cpu_py = _root / "cpu" / "recommender.py"
_spec = importlib.util.spec_from_file_location("_cpu_recommender", _cpu_py)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

CpuUserKnnRecommender = _mod.CpuUserKnnRecommender


def _toy_df() -> pd.DataFrame:
    # U1 likes I1 and I2.
    # U2 similar to U1 and likes I3.
    # U3 mostly different.
    rows = [
        {"reviewerID": "U1", "asin": "I1", "rating": 5.0},
        {"reviewerID": "U1", "asin": "I2", "rating": 4.5},
        {"reviewerID": "U2", "asin": "I1", "rating": 5.0},
        {"reviewerID": "U2", "asin": "I2", "rating": 4.0},
        {"reviewerID": "U2", "asin": "I3", "rating": 5.0},
        {"reviewerID": "U3", "asin": "I4", "rating": 5.0},
    ]
    return pd.DataFrame(rows)


def test_cpu_recommender_recommend_for_known_user():
    model = CpuUserKnnRecommender.from_dataframe(_toy_df())
    recs = model.recommend_for_user("U1", top_k=3, n_neighbors=2)
    assert isinstance(recs, list)
    assert "I3" in recs  # picked from similar user U2
    assert "I1" not in recs and "I2" not in recs  # exclude seen defaults to True


def test_cpu_recommender_unknown_user_returns_empty():
    model = CpuUserKnnRecommender.from_dataframe(_toy_df())
    assert model.recommend_for_user("NEW_USER") == []

