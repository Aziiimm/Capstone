import cudf
import cupy as cp

class AmazonRecommenderGPU:
    def __init__(self, knn_model, sparse_matrix, title_map):
        self.model = knn_model
        self.sparse_matrix = sparse_matrix 
        self.title_map = title_map

    def recommend(self, user_history_indices: list, top_k: int = 10):
        """
        Input: List of integers (item_indices the user already bought)
        """
        query_indices = cp.array(user_history_indices)

        distances, indices = self.model.kneighbors(self.sparse_matrix[query_indices])

        flat_indices = indices.ravel()
        flat_distances = distances.ravel()
        
        final_indices = flat_indices.get() 
        
        return self.format_results(final_indices, top_k)
    
    def format_results(self, indices, top_k):
            # We skip the first one because k-NN always finds the item itself as the 1st neighbor
            results = []
            for idx in indices[1:top_k+1]:
                title = self.title_map.get(int(idx), "Unknown Product")
                results.append(title)
            return results