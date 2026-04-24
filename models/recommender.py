import cudf
import cupy as cp


class AmazonRecommenderGPU:
    def __init__(self, knn_model, sparse_matrix, title_map, asin_to_idx=None):
        self.model = knn_model
        self.sparse_matrix = sparse_matrix
        self.title_map = title_map
        # asin → item_idx; empty dict keeps old pickles loadable without retraining
        self.asin_to_idx = asin_to_idx or {}

    @property
    def idx_to_asin(self):
        """Reverse map built once on first access."""
        if not hasattr(self, "_idx_to_asin"):
            self._idx_to_asin = {v: k for k, v in self.asin_to_idx.items()}
        return self._idx_to_asin

    def recommend(self, user_history_indices: list, top_k: int = 10):
        """
        Input: List of integers (item_indices the user already interacted with).
        Returns: list of {"asin": ..., "title": ...} dicts, length <= top_k.
        """
        # Cap neighbors so we never ask for more than the catalogue contains
        n_items = self.sparse_matrix.shape[0]
        n_to_fetch = min(top_k + len(user_history_indices), n_items)

        # Bug fix 1: cupyx sparse CSR matrices do not support CuPy-array fancy
        # indexing — pass a plain Python list so numpy-style row selection is used.
        row_data = self.sparse_matrix[user_history_indices]
        distances, indices = self.model.kneighbors(row_data, n_neighbors=n_to_fetch)

        # Transfer to CPU and flatten
        flat_indices = indices.ravel().get().tolist()

        # Bug fix 2: instead of blindly skipping index 0 (which assumed the query
        # item is always its own nearest neighbor), filter every input item out
        # regardless of where it appears in the result list.
        seen = set(user_history_indices)
        unique_results = []
        for idx in flat_indices:
            if idx not in seen:
                seen.add(idx)
                unique_results.append(idx)
            if len(unique_results) >= top_k:
                break

        return self.format_results(unique_results, top_k)

    def format_results(self, indices, top_k):
        results = []
        for idx in indices[:top_k]:
            results.append({
                "asin": self.idx_to_asin.get(int(idx), ""),
                "title": self.title_map.get(int(idx), "Unknown Product"),
            })
        return results
