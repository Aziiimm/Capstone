"""
Amazon Recommender API.
"""

from __future__ import annotations

import logging
import pickle

import pandas as pd

log = logging.getLogger(__name__)

# Amazon Recommender API: 
# serves ranked product recommendations using a trained SVD model
# Falls back to cold-start popularity ranking for new users
class AmazonRecommender:
    def __init__(self, svd_model, df: pd.DataFrame, category: str):
        self.svd = svd_model
        self.category = category

        self.all_items: list[str] = df["asin"].unique().tolist()
        self.user_items: dict[str, set[str]] = (
            df.groupby("reviewerID")["asin"].apply(set).to_dict()
        )

        agg_cols: dict = {
            "avg_hybrid_score": ("hybrid_score", "mean"),
            "avg_rating": ("rating", "mean"),
            "review_count": ("rating", "count"),
        }
        if "sentiment_compound" in df.columns:
            agg_cols["avg_sentiment"] = ("sentiment_compound", "mean")
        if "product_title" in df.columns:
            agg_cols["product_title"] = ("product_title", "first")

        agg = df.groupby("asin").agg(**agg_cols).reset_index()
        self._cold_start_pool = (
            agg[agg["review_count"] >= 20]
               .sort_values("avg_hybrid_score", ascending=False)
               .reset_index(drop=True)
        )

        self._title_map: dict[str, str] = {}
        if "product_title" in df.columns:
            self._title_map = (
                df.drop_duplicates("asin")
                  .set_index("asin")["product_title"]
                  .to_dict()
            )

    def recommend(self, user_id: str, top_k: int = 10) -> pd.DataFrame:
        """
        Return a ranked DataFrame of top-K product recommendations.

        Known user  → SVD predicts scores for all unseen items
        New user    → Cold-start popularity ranking
        """
        rated = self.user_items.get(user_id, set())
        if not rated:
            log.info("Cold-start path: user '%s' has no history", user_id)
            return self._cold_start(top_k)
        return self._svd_recommend(user_id, rated, top_k)

    def _svd_recommend(self, user_id: str, rated: set, top_k: int) -> pd.DataFrame:
        candidates = [iid for iid in self.all_items if iid not in rated]
        preds = [(iid, self.svd.predict(user_id, iid).est) for iid in candidates]
        preds.sort(key=lambda x: x[1], reverse=True)
        rows = [
            {
                "rank": rank,
                "asin": asin,
                "product_title": self._title_map.get(asin, "—"),
                "predicted_score": round(score, 4),
                "source": "SVD",
            }
            for rank, (asin, score) in enumerate(preds[:top_k], 1)
        ]
        return pd.DataFrame(rows)

    def _cold_start(self, top_k: int) -> pd.DataFrame:
        pool = self._cold_start_pool.head(top_k).copy()
        pool.insert(0, "rank", range(1, len(pool) + 1))
        pool["source"] = "cold_start"
        pool = pool.rename(columns={"avg_hybrid_score": "predicted_score"})
        keep = ["rank", "asin", "product_title", "predicted_score", "source"]
        return pool[[c for c in keep if c in pool.columns]]

    def save(self, prefix: str) -> None:
        path = f"{prefix}_recommender.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("AmazonRecommender saved → %s", path)

    @classmethod
    def load(cls, prefix: str) -> "AmazonRecommender":
        path = f"{prefix}_recommender.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
