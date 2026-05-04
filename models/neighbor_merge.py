"""
Merge kNN neighbor lists from multiple history items by summing similarity scores.

Sklearn/cuML cosine distance d is in [0, 2] for typical sparse rows; we use
score += max(0, 1 - d) per (history item -> candidate item). History indices
are excluded from the final ranking.
"""
from __future__ import annotations

import numpy as np


def merge_knn_scores(
    distances: np.ndarray,
    indices: np.ndarray,
    history_item_indices: np.ndarray,
    top_k: int,
) -> list[int]:
    """
    distances, indices: shape (n_queries, n_neighbors) from kneighbors on sparse_matrix[history].
    history_item_indices: shape (n_queries,) — row i corresponds to query item index history_item_indices[i].
    """
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    hist = np.asarray(history_item_indices, dtype=np.int64).reshape(-1)
    if distances.shape != indices.shape:
        raise ValueError("distances and indices must have the same shape")
    if distances.shape[0] != hist.size:
        raise ValueError("number of query rows must match len(history_item_indices)")

    scores: dict[int, float] = {}
    nq, nk = indices.shape
    for row in range(nq):
        qi = int(hist[row])
        for col in range(nk):
            j = int(indices[row, col])
            if j == qi:
                continue
            d = float(distances[row, col])
            sim = max(0.0, 1.0 - d)
            scores[j] = scores.get(j, 0.0) + sim

    for hi in hist:
        scores.pop(int(hi), None)

    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return ranked[:top_k]
