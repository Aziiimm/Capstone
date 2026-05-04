"""CPU item-item kNN recommender (sklearn + SciPy CSR). Pickle expects this module name."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from neighbor_merge import merge_knn_scores


class AmazonRecommenderCPU:
    def __init__(
        self,
        knn_model: NearestNeighbors,
        sparse_matrix: sparse.csr_matrix,
        title_map: dict[int, str],
    ):
        self.model = knn_model
        self.sparse_matrix = sparse_matrix
        self.title_map = title_map

    def recommend(self, user_history_indices: list[int], top_k: int = 10) -> list[str]:
        q = np.asarray(user_history_indices, dtype=np.int64)
        if q.size == 0:
            return []
        distances, indices = self.model.kneighbors(self.sparse_matrix[q])
        best_idx = merge_knn_scores(distances, indices, q, top_k)
        return [self.title_map.get(int(i), "Unknown Product") for i in best_idx]
