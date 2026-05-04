import argparse
import os
import pickle
import sys
import time
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

# Register classes for pickle (CPU and GPU pickles)
try:
    from recommender_cpu import AmazonRecommenderCPU  # noqa: F401
except ImportError:
    pass
try:
    from recommender import AmazonRecommenderGPU  # noqa: F401
except ImportError:
    pass


def _default_model_path() -> str:
    env = os.environ.get("MODEL_PATH")
    if env and os.path.isfile(env):
        return env
    for name in ("dev_recommender.pkl", "full_recommender.pkl"):
        p = _MODEL_DIR / name
        if p.is_file():
            return str(p)
    return str(_MODEL_DIR / "full_recommender.pkl")


def search_product(title_map: dict[int, str], query: str) -> list[tuple[int, str]]:
    """Match substring in title (case-insensitive), or exact numeric item index."""
    q = query.strip().lower()
    if not q:
        return []
    hits = [(idx, title) for idx, title in title_map.items() if q in title.lower()]
    if hits:
        return sorted(hits, key=lambda x: (str(x[1]).lower(), x[0]))
    if q.isdigit():
        idx = int(q)
        if idx in title_map:
            return [(idx, title_map[idx])]
    return []


def _catalog_search_hint(title_map: dict[int, str]) -> str | None:
    sample = next(iter(title_map.values()), "")
    s = str(sample).lower()
    if s.startswith("product b"):
        hi = max(title_map.keys(), default=0)
        return (
            "This pickle looks like synthetic titles (e.g. 'Product B00123'). "
            f"Try: product | b000 | b001 | or item ID 0–{hi}."
        )
    return None


def main():
    parser = argparse.ArgumentParser(description="Interactive similar-item recommendations.")
    parser.add_argument(
        "--model",
        default=_default_model_path(),
        help="Path to pickled recommender (default: dev_recommender.pkl or full_recommender.pkl next to this script, or MODEL_PATH env)",
    )
    args = parser.parse_args()
    model_path = args.model

    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        print(f"Hint: place a .pkl in {_MODEL_DIR} or set MODEL_PATH, or pass --model /path/to.pkl")
        return

    print(f"Loading recommender: {model_path}")
    load_start = time.perf_counter()
    with open(model_path, "rb") as f:
        recommender = pickle.load(f)
    print(f"Loaded in {time.perf_counter() - load_start:.2f}s")
    hint = _catalog_search_hint(recommender.title_map)
    if hint:
        print(hint)

    while True:
        print("\n" + "=" * 50)
        query = input("Search for a product (or 'q' to quit): ")

        if query.lower() == "q":
            break

        matches = search_product(recommender.title_map, query)

        if not matches:
            print("No products found. Use a substring from the title, or a numeric item ID (see hint above if shown).")
            continue

        print(f"\nFound {len(matches)} matches. Top 5:")
        top = matches[:5]
        for i, (idx, title) in enumerate(top):
            print(f" [{i}] {title} (item ID: {idx})")

        if len(top) == 1:
            choice_raw = input(
                "\nOnly one match — press Enter to use it, or type q to cancel: "
            ).strip()
            if choice_raw.lower() == "q":
                continue
            pick = 0
        else:
            choice_raw = input(
                "\nPick the line number in [brackets] (0–"
                f"{len(top) - 1}), not the title — or q to cancel: "
            ).strip()
            if choice_raw.lower() == "q":
                continue
            pick = int(choice_raw)

        try:
            target_idx = top[pick][0]
            print(f"Recommendations for: {recommender.title_map[target_idx]}...")

            inf_start = time.perf_counter()
            recommendations = recommender.recommend([target_idx], top_k=5)
            inf_time = (time.perf_counter() - inf_start) * 1000

            print(f"\nTop 5 similar items ({inf_time:.2f} ms):")
            for r in recommendations:
                print(f"  - {r}")

        except (ValueError, IndexError, KeyError):
            print(
                "Invalid selection. Use the bracket number only "
                f"(0–{len(top) - 1}), e.g. 0 for the first line."
            )


if __name__ == "__main__":
    main()
