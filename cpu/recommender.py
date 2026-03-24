from __future__ import annotations

"""
Week 4 CPU baseline recommender (user-based KNN style).

Pure pandas/numpy implementation so the baseline runs in any standard CPU
environment without extra ML dependencies.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def _cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between all row vectors in x."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    x_norm = x / norms
    return x_norm @ x_norm.T


@dataclass
class CpuUserKnnRecommender:
    user_item_matrix: np.ndarray
    similarity: np.ndarray
    user_to_idx: Dict[str, int]
    idx_to_item: Dict[int, str]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        score_col: str = "rating",
    ) -> "CpuUserKnnRecommender":
        """
        Build model from interactions DataFrame.

        Required columns: reviewerID, asin, and score_col (default: rating).
        """
        needed = {"reviewerID", "asin", score_col}
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        pivot = df.pivot_table(
            index="reviewerID",
            columns="asin",
            values=score_col,
            aggfunc="mean",
            fill_value=0.0,
        )

        user_ids = [str(i) for i in pivot.index.tolist()]
        item_ids = [str(i) for i in pivot.columns.tolist()]
        mat = pivot.to_numpy(dtype=float)
        sim = _cosine_similarity_matrix(mat)

        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        idx_to_item = {i: iid for i, iid in enumerate(item_ids)}

        return cls(
            user_item_matrix=mat,
            similarity=sim,
            user_to_idx=user_to_idx,
            idx_to_item=idx_to_item,
        )

    @classmethod
    def from_parquet(
        cls,
        path: str,
        score_col: str = "rating",
    ) -> "CpuUserKnnRecommender":
        df = pd.read_parquet(path)
        return cls.from_dataframe(df, score_col=score_col)

    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = 10,
        n_neighbors: int = 20,
        exclude_seen: bool = True,
    ) -> List[str]:
        """Recommend top-K items by neighbor-weighted score."""
        idx = self.user_to_idx.get(str(user_id))
        if idx is None:
            return []

        sims = self.similarity[idx].copy()
        sims[idx] = -np.inf  # remove self
        neighbor_order = np.argsort(sims)[::-1]
        neighbor_order = [i for i in neighbor_order if np.isfinite(sims[i])][:n_neighbors]
        if not neighbor_order:
            return []

        neighbor_weights = sims[neighbor_order]
        neighbor_matrix = self.user_item_matrix[neighbor_order]
        denom = np.sum(np.abs(neighbor_weights))
        if denom == 0:
            return []
        scores = (neighbor_weights @ neighbor_matrix) / denom

        if exclude_seen:
            seen_mask = self.user_item_matrix[idx] > 0
            scores = np.where(seen_mask, -np.inf, scores)

        ranked = np.argsort(scores)[::-1]
        ranked = [i for i in ranked if np.isfinite(scores[i])][:top_k]
        return [self.idx_to_item[i] for i in ranked]


def batch_recommend(
    model: CpuUserKnnRecommender,
    user_ids: Iterable[str],
    top_k: int = 10,
    n_neighbors: int = 20,
) -> Dict[str, List[str]]:
    """Recommend for a list of user IDs."""
    return {
        uid: model.recommend_for_user(uid, top_k=top_k, n_neighbors=n_neighbors)
        for uid in user_ids
    }

