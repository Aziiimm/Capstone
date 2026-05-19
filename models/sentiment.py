"""
Minimal sentiment scoring helper.

Prefers VADER if available (pip package: vaderSentiment). If not installed,
falls back to a tiny rule-based word list so the rest of the project still runs.

Returns a compound score in [-1, 1].
"""

from __future__ import annotations

import re
from functools import lru_cache

_WORD_RE = re.compile(r"[A-Za-z']+")


try:  # Optional dependency
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

    _VADER = SentimentIntensityAnalyzer()

    @lru_cache(maxsize=200_000)
    def compound_sentiment(text: str) -> float:
        if not text:
            return 0.0
        return float(_VADER.polarity_scores(text).get("compound", 0.0))

except Exception:  # pragma: no cover
    _POS = {
        "good",
        "great",
        "excellent",
        "amazing",
        "love",
        "loved",
        "perfect",
        "awesome",
        "nice",
        "best",
        "recommend",
        "recommended",
        "happy",
        "satisfied",
        "works",
        "working",
    }
    _NEG = {
        "bad",
        "terrible",
        "awful",
        "hate",
        "hated",
        "broken",
        "broke",
        "worst",
        "poor",
        "disappointed",
        "disappointing",
        "refund",
        "waste",
        "problem",
        "issues",
        "issue",
        "fail",
        "failed",
    }

    @lru_cache(maxsize=200_000)
    def compound_sentiment(text: str) -> float:
        if not text:
            return 0.0
        words = [w.lower() for w in _WORD_RE.findall(text)]
        if not words:
            return 0.0
        pos = sum(1 for w in words if w in _POS)
        neg = sum(1 for w in words if w in _NEG)
        raw = pos - neg
        # squash to [-1, 1]
        return max(-1.0, min(1.0, raw / 6.0))

