import argparse
import os
import pickle
import sys
import time
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

# Ensure pickle can resolve classes saved from recommender_cpu / recommender
try:
    from recommender_cpu import AmazonRecommenderCPU  # noqa: F401
except ImportError:
    pass
try:
    from recommender import AmazonRecommenderGPU  # noqa: F401
except ImportError:
    pass


def search_product(title_map, query):
    """Finds item_indices for products matching the search string."""
    query = query.lower()
    matches = [(idx, title) for idx, title in title_map.items() if query in title.lower()]
    return matches


def main():
    parser = argparse.ArgumentParser(description="Interactive similar-item recommendations.")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "models/full_recommender.pkl"),
        help="Pickled AmazonRecommenderCPU or AmazonRecommenderGPU",
    )
    args = parser.parse_args()
    model_path = args.model

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading recommender ({model_path})...")
    load_start = time.perf_counter()
    with open(model_path, "rb") as f:
        recommender = pickle.load(f)
    print(f"Loaded in {time.perf_counter() - load_start:.2f}s")

    while True:
        print("\n" + "=" * 50)
        query = input("Search for a product (or 'q' to quit): ")

        if query.lower() == "q":
            break

        matches = search_product(recommender.title_map, query)

        if not matches:
            print("No products found. Try a different keyword.")
            continue

        print(f"\nFound {len(matches)} matches. Top 5:")
        for i, (idx, title) in enumerate(matches[:5]):
            print(f" [{i}] {title} (ID: {idx})")

        choice = input("\nSelect a number to get recommendations: ")
        try:
            target_idx = matches[int(choice)][0]
            print(f"Generating recommendations for: {recommender.title_map[target_idx]}...")

            inf_start = time.perf_counter()
            recommendations = recommender.recommend([target_idx], top_k=5)
            inf_time = (time.perf_counter() - inf_start) * 1000

            print(f"\nTop 5 similar items (inference {inf_time:.2f} ms):")
            for r in recommendations:
                print(f"  - {r}")

        except (ValueError, IndexError, KeyError):
            print("Invalid selection. Enter a number from the list.")


if __name__ == "__main__":
    main()
