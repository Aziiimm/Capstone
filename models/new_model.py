import cudf
import rmm
import pickle
import os
import argparse
import glob
import pandas as pd
from cuml.neighbors import NearestNeighbors
from cupyx.scipy.sparse import coo_matrix
from recommender import AmazonRecommenderGPU

def main():
    parser = argparse.ArgumentParser(description="Train cuML Recommender using Integer-First strategy.")
    parser.add_argument("--path", type=str, required=True, help="Path with wildcard (e.g., 'output/*.parquet')")
    parser.add_argument("--neighbors", type=int, default=5, help="Number of neighbors for the model")
    parser.add_argument("--output", type=str, default="models/full_recommender.pkl", help="Save path")
    args = parser.parse_args()

    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=int(20e9), 
        managed_memory=True
    )

    print(f"Resolving path: {args.path}")
    input_dirs = glob.glob(args.path)
    
    if not input_dirs:
        print(f"Error: No directories found matching '{args.path}'.")
        return

    # Phase 1: Build Global Unique Sets for User and Item IDs
    print("Phase 1: Building global mapping for Reviewers and ASINs...")
    all_reviewers = []
    all_asins = []
    all_titles_cpu = []

    for d in input_dirs:
        print(f"  Scanning directory for unique IDs: {d}")
        # Only load the columns we need to build the map to save VRAM
        temp_df = cudf.read_parquet(d, columns=['reviewerID', 'asin', 'product_title'])
        
        all_reviewers.append(temp_df['reviewerID'].unique())
        all_asins.append(temp_df['asin'].unique())
        
        # Save titles to CPU now while we have the file open
        titles_subset = temp_df[['asin', 'product_title']].drop_duplicates().to_pandas()
        all_titles_cpu.append(titles_subset)
        
        del temp_df # Clear GPU memory for next file

    # Combine and get global unique codes
    unique_users = cudf.concat(all_reviewers).unique().reset_index(drop=True)
    global_user_map = cudf.DataFrame({'reviewerID': unique_users,
                                      'user_idx': cudf.Series(range(len(unique_users)), dtype='int32')})

    unique_items = cudf.concat(all_asins).unique().reset_index(drop=True)
    global_item_map = cudf.DataFrame({'asin': unique_items,
                                      'item_idx': cudf.Series(range(len(unique_items)), dtype='int32')})

    print(f"Found {len(global_user_map):,} unique users and {len(global_item_map):,} unique items.")

    # Phase 2: Load and Transform to Integers
    print("Phase 2: Converting data to integers and stacking...")
    dfs = []
    for d in input_dirs:
        print(f"  Processing: {d}")
        temp_df = cudf.read_parquet(d, columns=['reviewerID', 'asin', 'rating'])
        
        # Merge with our global maps to get the integer codes
        temp_df = temp_df.merge(global_user_map, on='reviewerID', how='left')
        temp_df = temp_df.merge(global_item_map, on='asin', how='left')
        
        # Drop the strings immediately!
        temp_df = temp_df.drop(columns=['reviewerID', 'asin'])
        dfs.append(temp_df)

    # Now concat will work because it's only numbers (int64/float32)
    df = cudf.concat(dfs)
    del dfs

    df['rating'] = df['rating'].astype('float32')
    df['item_idx'] = df['item_idx'].astype('int32')
    df['user_idx'] = df['user_idx'].astype('int32')

    # Phase 3: Build Matrix and Train
    n_users = len(global_user_map)
    n_items = len(global_item_map)

    print(f"Building sparse matrix: {n_items} items x {n_users} users...")
    sparse_matrix = coo_matrix(
        (df['rating'].values, (df['item_idx'].values, df['user_idx'].values)),
        shape=(n_items, n_users)
    ).tocsr()

    print(f"Training NearestNeighbors (k={args.neighbors})...")
    model = NearestNeighbors(n_neighbors=args.neighbors, metric='cosine')
    model.fit(sparse_matrix)

    # Phase 4: Final Mapping for the Pickle
    print("Building global title map...")
    full_titles_df = pd.concat(all_titles_cpu).drop_duplicates(subset=['asin'])
    # Merge titles with our global item index
    item_map_cpu = global_item_map.to_pandas()
    final_mapping = full_titles_df.merge(item_map_cpu, on='asin')
    title_map_dict = final_mapping.set_index('item_idx')['product_title'].to_dict()
    asin_to_idx_dict = final_mapping.set_index('asin')['item_idx'].to_dict()

    recommender = AmazonRecommenderGPU(model, sparse_matrix, title_map_dict, asin_to_idx_dict)
    
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
    with open(args.output, 'wb') as f:
        pickle.dump(recommender, f)

    print(f"SUCCESS: Global model saved to {args.output}")

if __name__ == "__main__":
    main()