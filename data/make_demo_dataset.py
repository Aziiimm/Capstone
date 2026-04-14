from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time
from typing import Any, Dict, Iterable, List


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_jsonl_gz(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_demo_rows(
    *,
    n_users: int,
    n_items: int,
    reviews_per_user: int,
    category: str,
    seed: int = 42,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create a tiny Amazon-Reviews-2023-shaped dataset that matches the loader expectations:
    - reviews use keys: user_id, asin, rating, text, timestamp, parent_asin
    - meta uses keys: parent_asin, title, main_category
    """
    rng = random.Random(seed)
    now = int(time.time())

    asins = [f"DEMO_ASIN_{i:04d}" for i in range(n_items)]
    parent_asins = [f"DEMO_PARENT_{i:04d}" for i in range(n_items)]
    asin_to_parent = dict(zip(asins, parent_asins))

    meta_rows: List[Dict[str, Any]] = []
    for asin, parent in zip(asins, parent_asins):
        meta_rows.append(
            {
                "parent_asin": parent,
                "title": f"Demo product for {asin}",
                "main_category": category,
            }
        )

    review_rows: List[Dict[str, Any]] = []
    for u in range(n_users):
        user_id = f"DEMO_USER_{u:04d}"
        # Ensure enough density to pass default filters (user>=6, item>=11) if you choose.
        choices = [rng.choice(asins) for _ in range(reviews_per_user)]
        for j, asin in enumerate(choices):
            rating = rng.choice([1, 2, 3, 4, 5])
            sentimentish = "great" if rating >= 4 else "ok" if rating == 3 else "bad"
            review_rows.append(
                {
                    "user_id": user_id,
                    "asin": asin,
                    "rating": float(rating),
                    "text": f"This is a {sentimentish} demo review for {asin}.",
                    "timestamp": now - (u * 1000 + j),
                    "parent_asin": asin_to_parent[asin],
                }
            )

    return review_rows, meta_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Create a tiny local demo dataset in dataset/.")
    p.add_argument("--category", default="Tools_and_Home_Improvement")
    p.add_argument("--out-dir", default="dataset")
    p.add_argument("--gz", action="store_true", help="Write .jsonl.gz instead of .jsonl")
    p.add_argument("--n-users", type=int, default=200)
    p.add_argument("--n-items", type=int, default=50)
    p.add_argument("--reviews-per-user", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    reviews, meta = make_demo_rows(
        n_users=args.n_users,
        n_items=args.n_items,
        reviews_per_user=args.reviews_per_user,
        category=args.category,
        seed=args.seed,
    )

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    review_name = f"{args.category}.jsonl" + (".gz" if args.gz else "")
    meta_name = f"meta_{args.category}.jsonl" + (".gz" if args.gz else "")
    review_path = os.path.join(out_dir, review_name)
    meta_path = os.path.join(out_dir, meta_name)

    if args.gz:
        _write_jsonl_gz(review_path, reviews)
        _write_jsonl_gz(meta_path, meta)
    else:
        _write_jsonl(review_path, reviews)
        _write_jsonl(meta_path, meta)

    print("Wrote demo files:")
    print(" reviews:", review_path)
    print(" meta:   ", meta_path)
    print(f" rows: reviews={len(reviews):,} meta={len(meta):,}")


if __name__ == "__main__":
    main()

