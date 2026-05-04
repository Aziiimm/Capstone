import cupy as cp

from neighbor_merge import merge_knn_scores


class AmazonRecommenderGPU:
    def __init__(self, knn_model, sparse_matrix, title_map):
        self.model = knn_model
        self.sparse_matrix = sparse_matrix 
        self.title_map = title_map

    def recommend(self, user_history_indices: list, top_k: int = 10):
        """
        Input: List of integers (item_indices the user already bought)
        """
        if not user_history_indices:
            return []
        query_indices = cp.asarray(user_history_indices)

        distances, indices = self.model.kneighbors(self.sparse_matrix[query_indices])

        dist_np = distances.get() if hasattr(distances, "get") else cp.asnumpy(distances)
        ind_np = indices.get() if hasattr(indices, "get") else cp.asnumpy(indices)
        hist_np = query_indices.get() if hasattr(query_indices, "get") else cp.asnumpy(query_indices)

        best_idx = merge_knn_scores(dist_np, ind_np, hist_np, top_k)
        return [self.title_map.get(int(i), "Unknown Product") for i in best_idx]