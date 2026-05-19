"""
TF-IDF content vectors over product titles for hybrid recommendations.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def fit_item_title_matrix(
    titles_by_item_idx: list[str],
    *,
    max_features: int = 4000,
    seed: int = 42,
) -> np.ndarray:
    """Return L2-normalized TF-IDF rows, shape (n_items, n_features). Missing titles → zero row."""
    cleaned = [
        (t if isinstance(t, str) and t.strip() else "unknown_item")
        for t in titles_by_item_idx
    ]
    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=1,
        ngram_range=(1, 2),
    )
    mat = vec.fit_transform(cleaned)
    return normalize(mat, norm="l2", axis=1).toarray().astype(np.float64)


def content_scores_from_history(
    title_matrix: np.ndarray,
    hist_idx: list[int],
    row_weights: np.ndarray | None,
    blocked: set[int],
) -> np.ndarray:
    """Cosine scores = title_matrix @ profile; profile = weighted mean of history rows."""
    if not hist_idx or title_matrix.size == 0:
        return np.zeros(title_matrix.shape[0], dtype=np.float64)
    hi = np.asarray(hist_idx, dtype=np.int64)
    if row_weights is not None:
        w = np.asarray(row_weights, dtype=np.float64).reshape(-1)
        if w.sum() <= 0.0:
            profile = title_matrix[hi].mean(axis=0)
        else:
            profile = (title_matrix[hi] * w[:, None]).sum(axis=0) / w.sum()
    else:
        profile = title_matrix[hi].mean(axis=0)
    pn = np.linalg.norm(profile)
    if pn == 0.0:
        return np.zeros(title_matrix.shape[0], dtype=np.float64)
    profile = profile / pn
    scores = title_matrix @ profile
    if blocked:
        scores[list(blocked)] = -np.inf
    return scores
