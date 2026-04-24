"""
Tests for filter_counts (user >= 6 reviews, item >= 11 reviews).
Uses Dask + Pandas (CPU path); fixture has known counts so we can assert output size.

Run from repo root: pytest tests/test_filter.py -v
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import dask.dataframe as dd

# Load data/filter.py by path so tests work even if "data" package is not on path
_root = Path(__file__).resolve().parent.parent
_filter_py = _root / "data" / "filter.py"
_spec = importlib.util.spec_from_file_location("_data_filter", _filter_py)
_filter_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_filter_mod)
filter_counts = _filter_mod.filter_counts


def _make_fixture_df() -> pd.DataFrame:
    """Small dataframe with known user/item counts for filter assertions.
    - U1: 6 reviews (all on I1) -> kept
    - U2: 5 reviews (all on I2) -> user dropped
    - U3: 7 reviews (5 on I1, 2 on I3) -> I3 has only 2 reviews -> item dropped, so U3's 2 rows on I3 dropped
    After filter: U1 (6) + U3 on I1 (5) = 11 rows. I1 has 11, I2 has 5 (dropped), I3 has 2 (dropped).
    """
    rows = []
    # U1: 6 x I1
    for _ in range(6):
        rows.append({"reviewerID": "U1", "asin": "I1", "rating": 4.0, "reviewText": "x", "timestamp": 1000})
    # U2: 5 x I2 (user U2 will be dropped)
    for _ in range(5):
        rows.append({"reviewerID": "U2", "asin": "I2", "rating": 4.0, "reviewText": "x", "timestamp": 1000})
    # U3: 5 x I1, 2 x I3 (item I3 will be dropped)
    for _ in range(5):
        rows.append({"reviewerID": "U3", "asin": "I1", "rating": 4.0, "reviewText": "x", "timestamp": 1000})
    for _ in range(2):
        rows.append({"reviewerID": "U3", "asin": "I3", "rating": 4.0, "reviewText": "x", "timestamp": 1000})
    return pd.DataFrame(rows)


@pytest.fixture
def dask_fixture():
    """Dask DataFrame (CPU path) with reviewerID and asin for filter tests."""
    pdf = _make_fixture_df()
    return dd.from_pandas(pdf, npartitions=2)


def test_filter_counts_dask(dask_fixture):
    """Filter keeps only users with >= 6 and items with >= 11 reviews (in original data)."""
    result = filter_counts(dask_fixture)
    n = result.shape[0].compute()
    assert n == 11, f"Expected 11 rows after filter, got {n}"

    pdf = result.compute()
    # After filter: only users who had >= 6 reviews and items that had >= 11 in original data
    # So we expect users {U1, U3} (U2 dropped) and item {I1} (I2, I3 dropped)
    assert set(pdf["reviewerID"].unique()) == {"U1", "U3"}, "U2 should be dropped (had 5 reviews)"
    assert set(pdf["asin"].unique()) == {"I1"}, "Only I1 should remain (I2 had 5, I3 had 2 reviews)"
    # In the result a user can have < 6 rows (e.g. U3 has 5 here) because we dropped their rows on I3
    assert len(pdf) == 11


def test_filter_counts_pandas():
    """Filter also works on in-memory Pandas DataFrame (same logic)."""
    pdf = _make_fixture_df()
    result = filter_counts(pdf)
    assert len(result) == 11
    assert result["reviewerID"].nunique() <= 2  # U2 dropped
    assert "I2" not in result["asin"].values or result[result["asin"] == "I2"].shape[0] == 0  # I2 dropped
