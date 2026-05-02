"""CPU item–item kNN recommender (sklearn + SciPy CSR). Mirrors AmazonRecommenderGPU recommend API."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


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
        """item indices the user interacted with → top_k product titles (same heuristic as GPU)."""
        query_indices = np.asarray(user_history_indices, dtype=np.int64)
        _, indices = self.model.kneighbors(self.sparse_matrix[query_indices])
        flat_indices = indices.ravel()
        return self.format_results(flat_indices, top_k)

    def format_results(self, indices: np.ndarray, top_k: int) -> list[str]:
        results: list[str] = []
        for idx in indices[1 : top_k + 1]:
            results.append(self.title_map.get(int(idx), "Unknown Product"))
        return results
