from __future__ import annotations

"""
Simple GPU-based collaborative filtering example using cuDF + cuML.

This is a *prototype* to show how the Capstone pipeline can plug into RAPIDS:

- Reads the cleaned Parquet produced by `data.run_pipeline`.
- Uses cuDF to keep data on the GPU.
- Uses cuML KNN on a dense user–item matrix to recommend similar items.

Notes
-----
- This is meant for **dev-sized** datasets (e.g. your 50k-row sample), not the full
  750GB corpus. For large-scale runs you would switch to a sparse representation.
- All heavy GPU dependencies are optional; if RAPIDS is not installed, importing
  this module will fail cleanly with a helpful error.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

try:  # Optional GPU stack
    import cudf
    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    HAS_RAPIDS = True
except Exception:  # pragma: no cover - exercised only on GPU machines
    cudf = None  # type: ignore[assignment]
    cp = None  # type: ignore[assignment]
    NearestNeighbors = None  # type: ignore[assignment]
    HAS_RAPIDS = False


class RapidsNotAvailableError(RuntimeError):
    """Raised when a GPU/cuML feature is used without RAPIDS installed."""


def _require_rapids() -> None:
    if not HAS_RAPIDS:
        raise RapidsNotAvailableError(
            "RAPIDS (cudf + cuml) is not available. "
            "Install matching CUDA wheels, e.g.: "
            "`pip install cudf-cu12 dask-cudf-cu12 cuml-cu12` "
            "or use the official RAPIDS conda images."
        )


def load_interactions_parquet(path: str) -> "cudf.DataFrame":
    """
    Load interactions from a Parquet file written by `data.run_pipeline` using cuDF.

    Parameters
    ----------
    path:
        Path to a Parquet file (or directory) containing at least the columns
        `reviewerID`, `asin`, and `rating`.
    """
    _require_rapids()
    gdf = cudf.read_parquet(path)
    needed = ["reviewerID", "asin", "rating"]
    missing = [c for c in needed if c not in gdf.columns]
    if missing:
        raise ValueError(
            f"Parquet is missing required columns {missing}. "
            f"Found columns: {list(gdf.columns)}"
        )
    return gdf[needed]


@dataclass
class GpuKnnRecommender:
    """
    Very simple user-based KNN recommender on GPU using cuML.

    Workflow
    --------
    1. Call `GpuKnnRecommender.from_parquet(...)` to build the model from a Parquet
       file produced by the CPU/GPU data-cleaning pipeline.
    2. Call `recommend_for_user(user_id)` to get top-N item IDs for that user.

    This uses a **dense** user–item matrix on the GPU; use only for dev-sized data.
    """

    user_item_matrix: "cp.ndarray"
    user_to_idx: Dict[str, int]
    idx_to_item: Dict[int, str]
    knn: "NearestNeighbors"

    @classmethod
    def from_parquet(
        cls,
        path: str,
        min_rating: float = 1.0,
        n_neighbors: int = 20,
        metric: str = "cosine",
    ) -> "GpuKnnRecommender":
        """
        Build a GPU KNN model from a Parquet file of interactions.

        Parameters
        ----------
        path:
            Parquet path written by `data.run_pipeline`.
        min_rating:
            Minimum rating value to count as an interaction.
        n_neighbors:
            Number of neighbor users to search in cuML.
        metric:
            Distance metric for cuML NearestNeighbors (e.g. 'cosine', 'euclidean').
        """
        _require_rapids()
        gdf = load_interactions_parquet(path)
        gdf = gdf[gdf["rating"] >= min_rating]

        # Map arbitrary user/item IDs to contiguous integer indices using cuDF categories.
        gdf["user_code"] = gdf["reviewerID"].astype("category").cat.codes
        gdf["item_code"] = gdf["asin"].astype("category").cat.codes

        n_users = int(gdf["user_code"].max()) + 1
        n_items = int(gdf["item_code"].max()) + 1

        # Dense user–item matrix (n_users x n_items) on GPU.
        mat = cp.zeros((n_users, n_items), dtype=cp.float32)

        # Scatter ratings into the dense matrix.
        # Using .values gives us GPU arrays for fancy indexing.
        user_idx = gdf["user_code"].values
        item_idx = gdf["item_code"].values
        ratings = gdf["rating"].astype("float32").values
        mat[user_idx, item_idx] = ratings

        # Fit cuML KNN on the user vectors.
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        knn.fit(mat)

        # Build Python mappings for external IDs <-> indices (host-side dicts).
        # The .cat.categories is a GPU Series; convert to host to store in dicts.
        user_cats = gdf[["reviewerID", "user_code"]].drop_duplicates().to_pandas()
        item_cats = gdf[["asin", "item_code"]].drop_duplicates().to_pandas()

        user_to_idx = {
            str(row["reviewerID"]): int(row["user_code"])
            for _, row in user_cats.iterrows()
        }
        idx_to_item = {
            int(row["item_code"]): str(row["asin"])
            for _, row in item_cats.iterrows()
        }

        return cls(
            user_item_matrix=mat,
            user_to_idx=user_to_idx,
            idx_to_item=idx_to_item,
            knn=knn,
        )

    def _get_user_index(self, user_id: str) -> Optional[int]:
        return self.user_to_idx.get(str(user_id))

    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> List[str]:
        """
        Recommend top-K items for a given user ID.

        Parameters
        ----------
        user_id:
            Raw `reviewerID` from the dataset.
        top_k:
            Number of items to return.
        exclude_seen:
            If True, do not recommend items the user has already interacted with.
        """
        _require_rapids()
        idx = self._get_user_index(user_id)
        if idx is None:
            # Cold-start user; no history in this matrix.
            return []

        user_vec = self.user_item_matrix[idx : idx + 1]
        # Get neighbor users (including the user itself at position 0).
        _distances, indices = self.knn.kneighbors(user_vec)
        neighbor_indices = indices[0]

        # Drop self-neighbor (assumed to be at index 0).
        neighbor_indices = neighbor_indices[neighbor_indices != idx]
        if neighbor_indices.size == 0:
            return []

        neighbor_ratings = self.user_item_matrix[neighbor_indices]
        # Simple neighborhood aggregation: mean rating per item.
        scores = neighbor_ratings.mean(axis=0)

        if exclude_seen:
            seen_mask = self.user_item_matrix[idx] > 0
            scores = cp.where(seen_mask, -cp.inf, scores)

        # Get top-K item indices by score.
        score_np = cp.asnumpy(scores)
        top_idx = np.argsort(score_np)  # ascending
        # Filter out items with -inf scores, then take top_k in descending order.
        top_idx = [int(i) for i in top_idx if np.isfinite(score_np[i])][-top_k:][::-1]

        return [self.idx_to_item[i] for i in top_idx]


def batch_recommend(
    model: GpuKnnRecommender,
    user_ids: Iterable[str],
    top_k: int = 10,
) -> Dict[str, List[str]]:
    """
    Convenience helper: recommend for many users at once.

    Returns
    -------
    dict
        Mapping from `reviewerID` -> list of recommended `asin`s.
    """
    return {uid: model.recommend_for_user(uid, top_k=top_k) for uid in user_ids}

