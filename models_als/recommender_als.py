"""
ALS-based recommender. Wraps an implicit.als.AlternatingLeastSquares model
and exposes the same recommend(history, top_k) interface as the legacy KNN
recommender so the FastAPI server can swap backends with one import change.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


class AmazonRecommenderALS:
    def __init__(
        self,
        model: AlternatingLeastSquares,
        user_item: sp.csr_matrix,
        title_map: Dict[int, str],
        asin_to_idx: Dict[str, int],
    ):
        self.model = model
        self.user_item = user_item.tocsr()
        self.title_map = title_map
        self.asin_to_idx = asin_to_idx
        self._idx_to_asin: Dict[int, str] | None = None

    @property
    def idx_to_asin(self) -> Dict[int, str]:
        if self._idx_to_asin is None:
            self._idx_to_asin = {v: k for k, v in self.asin_to_idx.items()}
        return self._idx_to_asin

    @property
    def n_items(self) -> int:
        return self.user_item.shape[1]

    def recommend(self, user_history_indices: List[int], top_k: int = 10) -> List[Dict[str, str]]:
        """Synthesize a user from history and return top_k items they haven't seen."""
        history = [int(i) for i in user_history_indices if 0 <= int(i) < self.n_items]
        if not history:
            return []

        data = np.ones(len(history), dtype=np.float32)
        rows = np.zeros(len(history), dtype=np.int32)
        cols = np.array(history, dtype=np.int32)
        history_row = sp.csr_matrix((data, (rows, cols)), shape=(1, self.n_items))

        ids, _ = self.model.recommend(
            userid=0,
            user_items=history_row,
            N=top_k + len(history),
            recalculate_user=True,
            filter_already_liked_items=True,
        )

        seen = set(history)
        out: List[int] = []
        for idx in ids.tolist():
            idx = int(idx)
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
            if len(out) >= top_k:
                break

        return self._format(out)

    def similar_items(self, item_idx: int, top_k: int = 10) -> List[Dict[str, str]]:
        """Return items closest to item_idx in the learned factor space."""
        ids, _ = self.model.similar_items(int(item_idx), N=top_k + 1)
        out = [int(i) for i in ids.tolist() if int(i) != int(item_idx)]
        return self._format(out[:top_k])

    def _format(self, indices: List[int]) -> List[Dict[str, str]]:
        return [
            {
                "asin": self.idx_to_asin.get(idx, ""),
                "title": self.title_map.get(idx, "Unknown Product"),
            }
            for idx in indices
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path / "als_model.npz"))
        sp.save_npz(str(path / "user_item.npz"), self.user_item)
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(
                {"title_map": self.title_map, "asin_to_idx": self.asin_to_idx},
                f,
                protocol=4,
            )

    @classmethod
    def load(cls, path: str | Path) -> "AmazonRecommenderALS":
        path = Path(path)
        model = AlternatingLeastSquares.load(str(path / "als_model.npz"))
        user_item = sp.load_npz(str(path / "user_item.npz")).tocsr()
        with open(path / "metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        return cls(
            model=model,
            user_item=user_item,
            title_map=meta["title_map"],
            asin_to_idx=meta["asin_to_idx"],
        )
