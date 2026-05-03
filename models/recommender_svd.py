import pandas as pd
import cupy as cp
import logging
import pickle

log = logging.getLogger(__name__)


class AmazonRecommenderGPU:
    def __init__(self, model_dict: dict):
        self.knn          = model_dict['knn']
        self.latent_items = model_dict['latent_items']
        self.user_factors = model_dict['user_factors']
        self.asin_to_idx  = model_dict['asin_to_idx']
        self.idx_to_asin  = model_dict['idx_to_asin']
        self.idx_to_title = model_dict['idx_to_title']
        self.user_map     = model_dict['user_map']
        # FIX 1: Correctly load the history mapping
        self.user_items   = model_dict.get('user_items', {}) 
        self._cold_start_pool = model_dict.get('cold_start_df')

    def recommend(self, user_id: str, top_k: int = 10) -> pd.DataFrame:
        if user_id in self.user_map:        
            history = list(self.user_items.get(user_id, []))
            return self._user_factor_recommend(user_id, history, top_k)

        log.info("Cold-start path: user '%s' has no history", user_id)
        return self._cold_start(top_k)

    def _user_factor_recommend(self, reviewer_id: str,
                                history_asins: list[str], top_k: int) -> pd.DataFrame:
        """
        Uses the actual SVD user embedding — most accurate for known users.
        user_factors shape: (k, n_users)
        """
        u_idx       = self.user_map[reviewer_id]
        user_vec    = self.user_factors[:, u_idx]              # (k,)
        user_vec_n  = user_vec / (cp.linalg.norm(user_vec) + 1e-9)
        user_vec_n  = user_vec_n.reshape(1, -1)

        distances, neighbor_indices = self.knn.kneighbors(
            user_vec_n, n_neighbors=top_k + len(history_asins)
        )
        return self._format_results(
            neighbor_indices[0], distances[0],
            exclude_asins=set(history_asins),
            top_k=top_k, source="GPU-SVD-user"
        )

    def _item_average_recommend(self, history: list[str], top_k: int) -> pd.DataFrame:
        """
        For users not in the training set: average their history item vectors,
        re-normalize, then KNN search. Fixes the missing normalization bug.
        """
        indices    = [self.asin_to_idx[a] for a in history]
        user_vec   = self.latent_items[indices].mean(axis=0)   # (k,)
        user_vec_n = user_vec / (cp.linalg.norm(user_vec) + 1e-9)
        user_vec_n = user_vec_n.reshape(1, -1)

        distances, neighbor_indices = self.knn.kneighbors(
            user_vec_n, n_neighbors=top_k + len(history)
        )
        return self._format_results(
            neighbor_indices[0], distances[0],
            exclude_asins=set(history),
            top_k=top_k, source="GPU-SVD-item-avg"
        )

    def _format_results(self, neighbor_indices, distances,
                        exclude_asins: set, top_k: int, source: str) -> pd.DataFrame:
        rows = []
        rank = 1
        for i in range(len(neighbor_indices)):
            idx  = int(neighbor_indices[i])
            asin = self.idx_to_asin.get(idx)
            if asin in exclude_asins:
                continue
            rows.append({
                "rank":             rank,
                "asin":             asin,
                "product_title":    self.idx_to_title.get(idx, "—"),
                "similarity_score": round(1 - float(distances[i]), 4),
                "source":           source,
            })
            rank += 1
            if rank > top_k:
                break
        return pd.DataFrame(rows)

    def _cold_start(self, top_k: int) -> pd.DataFrame:
        if self._cold_start_pool is None:
            return pd.DataFrame([{"error": "Cold start pool not initialized"}])
        pool = self._cold_start_pool.head(top_k).copy()
        pool.insert(0, "rank", range(1, len(pool) + 1))
        pool["source"] = "cold_start"
        return pool

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "AmazonRecommenderGPU":
        with open(path, "rb") as f:
            return pickle.load(f)