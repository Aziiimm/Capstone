from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu.recommender import (
    GpuKnnRecommender,
    get_gpu_device_name,
    load_interactions_parquet,
    pick_any_reviewer_id,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU KNN recommender demo (RAPIDS cuDF/cuML).")
    p.add_argument("--parquet", required=True, help="Path to interactions parquet (file or directory).")
    p.add_argument("--user", default=None, help="reviewerID. If omitted, picks one automatically.")
    p.add_argument("--top-k", type=int, default=10, help="Number of recommendations.")
    p.add_argument("--neighbors", type=int, default=20, help="Number of neighbor users.")
    p.add_argument("--min-rating", type=float, default=1.0, help="Min rating to count as interaction.")
    p.add_argument("--metric", default="cosine", help="cuML KNN metric (e.g. cosine, euclidean).")
    p.add_argument("--timings-json", default=None, help="Optional path to write timings JSON.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    parquet_path = str(Path(args.parquet))

    print(f"GPU device: {get_gpu_device_name()}")

    build_timings: dict[str, float] = {}
    model = GpuKnnRecommender.from_parquet(
        parquet_path,
        min_rating=args.min_rating,
        n_neighbors=args.neighbors,
        metric=args.metric,
        timings=build_timings,
    )

    user_id = args.user
    if not user_id:
        gdf = load_interactions_parquet(parquet_path)
        user_id = pick_any_reviewer_id(gdf)

    if not user_id:
        raise SystemExit("Could not auto-pick a reviewerID from parquet.")

    rec_timings: dict[str, float] = {}
    recs = model.recommend_for_user(user_id, top_k=args.top_k, timings=rec_timings)

    print(f"User: {user_id}")
    print(f"Recommendations (top {args.top_k}):")
    for r in recs:
        print(f"- {r}")

    timings_out = {"build": build_timings, "recommend": rec_timings}
    print("Timings (s):")
    print(json.dumps(timings_out, indent=2, sort_keys=True))

    if args.timings_json:
        Path(args.timings_json).write_text(json.dumps(timings_out, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

