import cudf
import cupy as cp
import rmm
import argparse
import pickle
import time
import os
import glob
import numpy as np
from cupyx.scipy.sparse import csr_matrix
from cuml.neighbors import NearestNeighbors
from cupyx.scipy.sparse.linalg import svds

from recommender_svd import AmazonRecommenderGPU

def calculate_ranking_metrics(item_factors, user_factors, train_df, test_df, k=10, threshold=0.5, batch_size=512):
    """
    Vectorized ranking metrics with Training Set Masking.
    We penalize items already seen in training so they aren't recommended in the test phase.
    """
    # 1. Identify "Relevant" items in test set
    relevant_test = test_df[test_df['hybrid_score'] >= threshold]
    if len(relevant_test) == 0:
        return {"Error": f"No relevant items in test set with threshold {threshold}"}

    # 2. Build Ground Truth (Test Set)
    user_positives = (
        relevant_test.groupby('user_idx')['item_idx']
        .agg(list).to_pandas().to_dict()
    )
    
    # 3. Build Training History (To be masked)
    # We map user_idx -> list of item_idx they ALREADY saw in training
    train_interactions = (
        train_df.groupby('user_idx')['item_idx']
        .agg(list).to_pandas().to_dict()
    )
    
    test_user_indices = np.array(list(user_positives.keys()), dtype=np.int32)
    n_test_users = len(test_user_indices)

    all_precisions, all_recalls, all_hits, all_ndcgs = [], [], [], []

    for start in range(0, n_test_users, batch_size):
        batch_indices = test_user_indices[start : start + batch_size]

        # Compute raw scores for ALL items
        user_vecs = user_factors[:, batch_indices] # (k, batch)
        scores = item_factors @ user_vecs           # (n_items, batch)

        # 4. APPLY THE MASK
        # For each user in the batch, set their training items to -infinity
        for j, u_idx in enumerate(batch_indices):
            if u_idx in train_interactions:
                seen_items = train_interactions[u_idx]
                # This ensures the model only ranks NEW items for the evaluation
                scores[seen_items, j] = -1e9 

        # Now argsort will pick the top NEW items
        top_k_all = cp.argsort(scores, axis=0)[-k:][::-1]
        top_k_cpu = top_k_all.get()

        for j, u_idx in enumerate(batch_indices):
            actual = set(user_positives.get(u_idx, []))
            if not actual:
                continue
                
            top_k = top_k_cpu[:, j].tolist()
            hits_list = [1 if idx in actual else 0 for idx in top_k]
            num_hits = sum(hits_list)

            all_precisions.append(num_hits / k)
            all_recalls.append(num_hits / len(actual))
            all_hits.append(1 if num_hits > 0 else 0)

            # NDCG Calculation
            dcg  = sum(hits_list[i] / np.log2(i + 2) for i in range(len(hits_list)))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
            all_ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        f"Precision@{k}": float(np.mean(all_precisions)),
        f"Recall@{k}":    float(np.mean(all_recalls)),
        f"Hit_Rate@{k}":  float(np.mean(all_hits)),
        f"NDCG@{k}":      float(np.mean(all_ndcgs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",             type=str, required=True)
    parser.add_argument("--neighbors",        type=int, default=50)
    parser.add_argument("--output",           type=str, default="models/test_recommender_svd.pkl")
    parser.add_argument("--min_user_reviews", type=int, default=10)
    parser.add_argument("--min_item_reviews", type=int, default=20)
    parser.add_argument("--latent_comps",     type=int, default=200)
    parser.add_argument("--demo-user",        type=str, default=None) # Keep for quick testing
    args = parser.parse_args()

    rmm.reinitialize(pool_allocator=True, initial_pool_size=int(20e9))

    # Phase 1 & 2: Load and Merge all Parquet files
    print(f"Phase 1: Loading all Parquet files from {args.path}...")
    input_files = glob.glob(args.path)
    dfs = []
    for f in input_files:
        temp_df = cudf.read_parquet(f, columns=['reviewerID', 'asin', 'hybrid_score', 'product_title'])
        # Basic filtering per category to keep VRAM manageable
        item_counts = temp_df['asin'].value_counts()
        keep_items  = item_counts[item_counts >= args.min_item_reviews].index
        temp_df     = temp_df[temp_df['asin'].isin(keep_items)]
        dfs.append(temp_df)

    full_df = cudf.concat(dfs); del dfs
    
    # Global user filtering
    user_counts = full_df['reviewerID'].value_counts()
    keep_users  = user_counts[user_counts >= args.min_user_reviews].index
    full_df     = full_df[full_df['reviewerID'].isin(keep_users)]

    # --- NEW: Build the User History Map (Crucial for the "recommend" interface) ---
    print("Building user history mapping (CPU-side for serving)...")
    user_items = (
        full_df[['reviewerID', 'asin']]
        .to_pandas()
        .groupby('reviewerID')['asin']
        .apply(set)
        .to_dict()
    )

    # Phase 3: Index mapping + train/test split
    unique_users = full_df['reviewerID'].unique()
    user_map = cudf.DataFrame({
        'reviewerID': unique_users,
        'user_idx':   cp.arange(len(unique_users), dtype='int32')
    })
    unique_items = full_df['asin'].unique()
    item_map = cudf.DataFrame({
        'asin':     unique_items,
        'item_idx': cp.arange(len(unique_items), dtype='int32')
    })

    full_df  = full_df.merge(user_map, on='reviewerID').merge(item_map, on='asin')
    n_users  = len(user_map)
    n_items  = len(item_map)

    print("Phase 3: 80/20 split...")
    full_df  = full_df.sample(frac=1, random_state=42)
    split    = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split]
    test_df  = full_df.iloc[split:]

    # Phase 4: Sparse matrix
    print(f"Phase 4: Building sparse matrix ({n_items} items × {n_users} users)...")

    # Defensive clip before building — NaN/Inf in hybrid_score will crash the solver
    scores = train_df['hybrid_score'].values.astype('float32')
    scores = cp.clip(cp.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    sparse_mtx = csr_matrix(
        (scores,
        (train_df['item_idx'].values, train_df['user_idx'].values)),
        shape=(n_items, n_users)
    )

    # Drop zero-sum ROWS (items with no training interactions)
    row_sums   = cp.array(sparse_mtx.sum(axis=1)).flatten()
    valid_row_mask = row_sums > 0
    if not valid_row_mask.all():
        n_dropped = int((~valid_row_mask).sum())
        print(f"  Removing {n_dropped} empty item rows...")
        sparse_mtx      = sparse_mtx[valid_row_mask, :]
        active_item_map = item_map.iloc[valid_row_mask]
    else:
        active_item_map = item_map

    # Drop zero-sum COLUMNS (users with no training interactions after row filter)
    # This is what causes the ARPACK floating point exception
    col_sums = cp.array(sparse_mtx.sum(axis=0)).flatten()
    valid_col_mask = col_sums > 0
    if not valid_col_mask.all():
        n_dropped_cols = int((~valid_col_mask).sum())
        print(f"  Removing {n_dropped_cols} empty user columns...")
        sparse_mtx = sparse_mtx[:, valid_col_mask]

    print(f"  Clean matrix shape: {sparse_mtx.shape}")

    # Phase 4b: Randomized SVD (replaces ARPACK-based svds entirely)
    def randomized_svd_gpu(M, k, n_oversampling=10, n_power_iter=3, random_state=42):
        """
        Halko et al. 2009 randomized SVD.
        All ops are GPU-native — no ARPACK, no FPE risk.
        M: cupyx CSR sparse (n_items × n_users)
        Returns U (n_items, k), s (k,), Vt (k, n_users)
        """
        cp.random.seed(random_state)
        _, m = M.shape
        rank = k + n_oversampling

        # Stage A: Random projection + power iteration
        Omega = cp.random.randn(m, rank).astype('float32')
        Y = M @ Omega                           # (n_items, rank) — sparse @ dense, fine
        for _ in range(n_power_iter):
            Y = M @ (M.T @ Y)                   # power iteration improves accuracy
        Q, _ = cp.linalg.qr(Y)                 # (n_items, rank) orthonormal basis

        # Stage B: Project M into the low-dim subspace and factor
        B = Q.T @ M                             # (rank, n_users) — still sparse-aware
        U_hat, s, Vt = cp.linalg.svd(B.toarray() if hasattr(B, 'toarray') else B,
                                    full_matrices=False)
        U = Q @ U_hat                           # (n_items, rank)

        return U[:, :k], s[:k], Vt[:k, :]


    k_latent = min(args.latent_comps, sparse_mtx.shape[0] - 2, sparse_mtx.shape[1] - 2)
    print(f"Phase 4b: Randomized SVD into {k_latent} latent factors...")
    t0 = time.perf_counter()

    U, s, Vt = randomized_svd_gpu(sparse_mtx, k=k_latent)
    # Already in descending order — no reversal needed

    item_factors = U * s    # (n_items, k)
    user_factors = Vt       # (k, n_active_users)

    train_time = time.perf_counter() - t0
    print(f"SVD trained in {train_time:.2f}s")

# Phase 5: Vectorized ranking evaluation
    print("Phase 5: Calculating ranking metrics (K=10)...")
    valid_indices = cp.where(valid_row_mask)[0]
    
    # We need to filter train_df as well so the indices match the matrix
    train_filtered = train_df[train_df['item_idx'].isin(valid_indices)].copy()
    train_filtered['item_idx'] = cp.searchsorted(valid_indices, train_filtered['item_idx'].values)
    
    test_filtered = test_df[test_df['item_idx'].isin(valid_indices)].copy()
    test_filtered['item_idx'] = cp.searchsorted(valid_indices, test_filtered['item_idx'].values)

    # Pass BOTH train_filtered and test_filtered
    metrics = calculate_ranking_metrics(
        item_factors, user_factors, train_filtered, test_filtered, k=10, threshold=0.5
    )

    print("Ranking Metrics:", {k: f"{v:.4f}" for k, v in metrics.items()})

    # Phase 6: Build and save payload
    print("Building lookup tables and saving...")
    active_items_df = active_item_map.to_pandas().reset_index(drop=True)
    active_items_df['matrix_idx'] = active_items_df.index

    titles_lookup  = full_df[['asin', 'product_title']].drop_duplicates('asin').to_pandas()
    merged         = active_items_df.merge(titles_lookup, on='asin', how='left')

    # idx_to_asin and idx_to_title are the natural lookup direction
    idx_to_asin    = merged.set_index('matrix_idx')['asin'].to_dict()
    idx_to_title   = merged.set_index('matrix_idx')['product_title'].fillna("Unknown").to_dict()
    # asin_to_idx for the recommender's lookup direction
    asin_to_idx    = {v: k for k, v in idx_to_asin.items()}

    # Normalize item factors for cosine KNN
    norms              = cp.linalg.norm(item_factors, axis=1, keepdims=True) + 1e-9
    item_factors_norm  = item_factors / norms

    knn_model = NearestNeighbors(n_neighbors=args.neighbors, metric='cosine')
    knn_model.fit(item_factors_norm)

    # Build cold-start pool: top items by mean hybrid_score with enough reviews
    cold_start_df = (
        full_df[['asin', 'hybrid_score', 'product_title']]
        .to_pandas()
        .groupby(['asin', 'product_title'])
        ['hybrid_score'].agg(['mean', 'count'])
        .reset_index()
        .query('count >= 50')
        .sort_values('mean', ascending=False)
        .rename(columns={'mean': 'avg_score', 'count': 'n_reviews'})
        .reset_index(drop=True)
    )

    payload = {
        'knn':            knn_model,
        'latent_items':   item_factors_norm,
        'user_factors':   user_factors,
        'asin_to_idx':    asin_to_idx,
        'idx_to_asin':    idx_to_asin,
        'idx_to_title':   idx_to_title,
        'user_map':       user_map.to_pandas().set_index('reviewerID')['user_idx'].to_dict(),
        'user_items':     user_items, 
        'cold_start_df':  cold_start_df,
        'metrics':        metrics,
        'train_time':     train_time,
    }

    # Wrap the payload in the serving class
    recommender = AmazonRecommenderGPU(payload)
    
    # Save once using the class method
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    recommender.save(args.output) 
    print(f"Full Recommender Object saved to {args.output}")

    # Optional Demo - Now using the UserID String!
    if args.demo_user:
        try:
            recs = recommender.recommend(args.demo_user, top_k=10)
            print(f"\nTop-10 recommendations for '{args.demo_user}':")
            print(recs.to_string(index=False))
        except Exception as e:
            print(f"Demo lookup failed: {e}")

if __name__ == "__main__":
    main()