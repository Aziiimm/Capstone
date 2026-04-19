import pickle
import cupy as cp
import time
import os
# Ensure your class definition is available for the pickle loader
from recommender import AmazonRecommenderGPU 

def search_product(title_map, query):
    """Finds item_indices for products matching the search string."""
    query = query.lower()
    matches = [(idx, title) for idx, title in title_map.items() if query in title.lower()]
    return matches

def main():
    model_path = "models/full_recommender.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    # 1. Load the "Brain" into memory
    print(f"🧠 Loading Global Recommender ({model_path})...")
    load_start = time.perf_counter()
    with open(model_path, 'rb') as f:
        recommender = pickle.load(f)
    print(f"✅ Loaded in {time.perf_counter() - load_start:.2f}s")

    while True:
        print("\n" + "="*50)
        query = input("🔍 Search for a product (or 'q' to quit): ")
        
        if query.lower() == 'q':
            break

        # 2. Find the ID for the search term
        matches = search_product(recommender.title_map, query)
        
        if not matches:
            print("❓ No products found. Try a different keyword.")
            continue

        print(f"\nFound {len(matches)} matches. Top 5:")
        for i, (idx, title) in enumerate(matches[:5]):
            print(f" [{i}] {title} (ID: {idx})")

        choice = input("\nSelect a number to get recommendations: ")
        try:
            target_idx = matches[int(choice)][0]
            print(f"✨ Generating recommendations for: {recommender.title_map[target_idx]}...")

            # 3. GPU INFERENCE
            # We wrap the single index in a list [target_idx] as your class expects a history list
            inf_start = time.perf_counter()
            recommendations = recommender.recommend([target_idx], top_k=5)
            inf_time = (time.perf_counter() - inf_start) * 1000 # Convert to ms

            print(f"\n✅ Top 5 Similar Items (Inference took {inf_time:.2f}ms):")
            # Note: Your recommender.py returns the result of _format_results
            for r in recommendations:
                print(f" • {r}") # Adjust this depending on how your _format_results is written

        except (ValueError, IndexError):
            print("❌ Invalid selection. Please enter a number from the list.")

if __name__ == "__main__":
    main()