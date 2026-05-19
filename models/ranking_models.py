"""
Extra ranking backends for evaluate_ranking.py: implicit ALS, kNN+SVD ensemble.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from neighbor_merge import merge_knn_scores


def apply_als_confidence(
    user_item_csr: sparse.csr_matrix,
    alpha: float,
) -> sparse.csr_matrix:
    """implicit-style confidence: value = 1 + alpha * raw (for nonnegative inputs)."""
    if alpha <= 0.0:
        return user_item_csr
    m = user_item_csr.tocsr().astype(np.float32)
    coo = m.tocoo()
    conf = 1.0 + float(alpha) * np.maximum(coo.data, 0.0)
    return sparse.coo_matrix((conf, (coo.row, coo.col)), shape=m.shape).tocsr()


def fit_als(
    user_item_csr: sparse.csr_matrix,
    *,
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 15,
    seed: int = 42,
    confidence_alpha: float = 0.0,
):
    """Fit implicit ALS on CSR matrix shaped (n_users, n_items)."""
    try:
        import implicit
    except ImportError as e:
        raise ImportError(
            "ALS requires the 'implicit' package. Install: pip install implicit"
        ) from e

    n_users, n_items = user_item_csr.shape
    factors = max(1, min(factors, min(n_users, n_items) - 1))
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=seed,
        use_gpu=False,
    )
    ui = apply_als_confidence(user_item_csr, confidence_alpha).astype(np.float32)
    model.fit(ui)
    return model


def als_dense_scores(model, user_idx: int, n_items: int) -> np.ndarray:
    uf = model.user_factors[user_idx]
    sc = model.item_factors @ uf
    if sc.size < n_items:
        out = np.full(n_items, -np.inf, dtype=np.float64)
        out[: sc.size] = sc.astype(np.float64)
        return out
    return sc[:n_items].astype(np.float64)


def als_recommend_indices(
    model,
    user_idx: int,
    blocked: set[int],
    top_k: int,
    train_user_item_row: sparse.csr_matrix | None = None,
) -> list[int]:
    """Top-k item indices for one user; excludes blocked (already seen in train)."""
    n_items = model.item_factors.shape[0]
    ids, scores = model.recommend(
        user_idx,
        train_user_item_row,
        N=min(top_k + len(blocked) + 20, n_items),
        filter_already_liked_items=False,
    )
    out: list[int] = []
    for i, _ in zip(ids, scores):
        j = int(i)
        if j in blocked:
            continue
        out.append(j)
        if len(out) >= top_k:
            break
    return out


def fit_nmf_item_factors(
    mat_item_user: sparse.csr_matrix,
    n_components: int,
    seed: int,
) -> np.ndarray:
    max_components = max(1, min(mat_item_user.shape) - 1)
    comps = max(1, min(n_components, max_components))
    nmf = NMF(n_components=comps, init="nndsvda", random_state=seed, max_iter=400)
    factors = nmf.fit_transform(mat_item_user.astype(np.float64)).astype(np.float64)
    norms = np.linalg.norm(factors, axis=1)
    norms[norms == 0.0] = 1.0
    return factors / norms[:, None]


def fit_svd_item_factors(mat_item_user: sparse.csr_matrix, n_components: int, seed: int) -> np.ndarray:
    max_components = max(1, min(mat_item_user.shape) - 1)
    comps = max(1, min(n_components, max_components))
    svd = TruncatedSVD(n_components=comps, random_state=seed)
    factors = svd.fit_transform(mat_item_user).astype(np.float64)
    norms = np.linalg.norm(factors, axis=1)
    norms[norms == 0.0] = 1.0
    return factors / norms[:, None]


def svd_recommend_indices(
    item_factors: np.ndarray,
    hist_idx: list[int],
    blocked: set[int],
    top_k: int,
    row_weights: np.ndarray | None = None,
) -> list[int]:
    if not hist_idx:
        return []
    hi = np.asarray(hist_idx, dtype=np.int64)
    if row_weights is not None:
        w = np.asarray(row_weights, dtype=np.float64).reshape(-1)
        if w.sum() <= 0.0:
            u = item_factors[hi].mean(axis=0)
        else:
            u = (item_factors[hi] * w[:, None]).sum(axis=0) / w.sum()
    else:
        u = item_factors[hi].mean(axis=0)
    un = np.linalg.norm(u)
    if un == 0.0:
        return []
    u = u / un
    scores = item_factors @ u
    if blocked:
        scores[list(blocked)] = -np.inf
    k = min(top_k, scores.size - 1)
    if k <= 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top])].tolist()


def knn_score_dict(
    model: NearestNeighbors,
    mat: sparse.csr_matrix,
    history_item_idx: list[int],
    row_weights: np.ndarray | None = None,
    item_popularity: np.ndarray | None = None,
    popularity_alpha: float = 0.0,
    query_neighbors: int | None = None,
) -> dict[int, float]:
    """Same scoring as merge_knn_scores but returns all candidate scores."""
    if not history_item_idx:
        return {}
    q = np.asarray(history_item_idx, dtype=np.int64)
    n_query = query_neighbors if query_neighbors is not None else model.n_neighbors
    n_query = min(max(int(n_query), 1), max(1, mat.shape[0] - 1))
    distances, indices = model.kneighbors(mat[q], n_neighbors=n_query)
    nq = q.size
    if row_weights is None:
        rw = np.ones(nq, dtype=np.float64)
    else:
        rw = np.asarray(row_weights, dtype=np.float64).reshape(-1)
    scores: dict[int, float] = {}
    for row in range(nq):
        qi = int(q[row])
        w = float(rw[row])
        for col in range(indices.shape[1]):
            j = int(indices[row, col])
            if j == qi:
                continue
            d = float(distances[row, col])
            sim = max(0.0, 1.0 - d)
            scores[j] = scores.get(j, 0.0) + sim * w
    for hi in q:
        scores.pop(int(hi), None)
    if item_popularity is not None and popularity_alpha > 0.0:
        pop = np.asarray(item_popularity, dtype=np.float64)
        for j in list(scores.keys()):
            if j < pop.size:
                scores[j] += popularity_alpha * float(pop[j])
    return scores


def knn_recommend_from_scores(scores: dict[int, float], top_k: int) -> list[int]:
    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return ranked[:top_k]


def ensemble_recommend_indices(
    knn_scores: dict[int, float],
    svd_scores: np.ndarray,
    blocked: set[int],
    top_k: int,
    knn_weight: float = 0.5,
) -> list[int]:
    """Blend normalized kNN score dict with dense SVD cosine scores."""
    if svd_scores.size == 0 and not knn_scores:
        return []
    knn_w = float(np.clip(knn_weight, 0.0, 1.0))
    svd_w = 1.0 - knn_w

    def _norm_dict(d: dict[int, float]) -> dict[int, float]:
        if not d:
            return {}
        vals = list(d.values())
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            return {k: 1.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    knn_n = _norm_dict(knn_scores)
    svd_n = svd_scores.copy().astype(np.float64)
    if blocked:
        svd_n[list(blocked)] = -np.inf
    finite = svd_n[np.isfinite(svd_n)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        if hi > lo:
            svd_n = (svd_n - lo) / (hi - lo)
        else:
            svd_n = np.where(np.isfinite(svd_n), 1.0, -np.inf)

    candidates = set(knn_n.keys()) | set(range(svd_n.size))
    blended: dict[int, float] = {}
    for j in candidates:
        if j in blocked:
            continue
        ks = knn_n.get(j, 0.0)
        ss = float(svd_n[j]) if j < svd_n.size and np.isfinite(svd_n[j]) else 0.0
        blended[j] = knn_w * ks + svd_w * ss
    ranked = sorted(blended.keys(), key=lambda x: blended[x], reverse=True)
    return ranked[:top_k]


def _normalize_dense(scores: np.ndarray, blocked: set[int]) -> np.ndarray:
    s = scores.copy().astype(np.float64)
    if blocked:
        s[list(blocked)] = -np.inf
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        return s
    lo, hi = float(finite.min()), float(finite.max())
    if hi > lo:
        s = np.where(np.isfinite(s), (s - lo) / (hi - lo), -np.inf)
    else:
        s = np.where(np.isfinite(s), 1.0, -np.inf)
    return s


def _normalize_dict(d: dict[int, float]) -> dict[int, float]:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def stack_recommend_indices(
    score_sources: list[tuple[float, dict[int, float] | np.ndarray]],
    blocked: set[int],
    top_k: int,
    n_items: int,
) -> list[int]:
    """Weighted blend of multiple score sources (each dict or dense array)."""
    blended = np.zeros(n_items, dtype=np.float64)
    total_w = 0.0
    for weight, src in score_sources:
        w = float(weight)
        if w <= 0.0:
            continue
        total_w += w
        if isinstance(src, dict):
            norm = _normalize_dict(src)
            for j, v in norm.items():
                if j < n_items:
                    blended[j] += w * v
        else:
            dense = _normalize_dense(np.asarray(src), blocked)
            blended[: min(n_items, dense.size)] += w * dense[:n_items]
    if total_w > 0.0:
        blended /= total_w
    if blocked:
        blended[list(blocked)] = -np.inf
    k = min(top_k, int(np.isfinite(blended).sum()) - 1)
    if k <= 0:
        return []
    top = np.argpartition(-blended, k - 1)[:k]
    return top[np.argsort(-blended[top])].tolist()


def blend_with_content(
    cf_scores: dict[int, float] | np.ndarray,
    content_scores: np.ndarray,
    blocked: set[int],
    top_k: int,
    content_weight: float,
    n_items: int,
) -> list[int]:
    """Blend collaborative scores with content TF-IDF scores."""
    cw = float(np.clip(content_weight, 0.0, 1.0))
    cf_w = 1.0 - cw
    sources: list[tuple[float, dict[int, float] | np.ndarray]] = []
    if cf_w > 0.0:
        sources.append((cf_w, cf_scores))
    if cw > 0.0:
        sources.append((cw, content_scores))
    return stack_recommend_indices(sources, blocked, top_k, n_items)
