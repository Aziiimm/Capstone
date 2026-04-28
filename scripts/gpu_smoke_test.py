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
    p = argparse.ArgumentParser(description="GPU smoke test for RAPIDS-based recommender.")
    p.add_argument("--parquet", default="output/sample.parquet", help="Parquet path to test.")
    p.add_argument("--top-k", type=int, default=10, help="Number of recommendations.")
    p.add_argument("--neighbors", type=int, default=20, help="Number of neighbor users.")
    p.add_argument("--timings-json", default=None, help="Optional path to write timings JSON.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    parquet_path = str(Path(args.parquet))

    print(f"GPU device: {get_gpu_device_name()}")

    build_timings: dict[str, float] = {}
    model = GpuKnnRecommender.from_parquet(
        parquet_path,
        n_neighbors=args.neighbors,
        timings=build_timings,
    )

    gdf = load_interactions_parquet(parquet_path)
    user_id = pick_any_reviewer_id(gdf)
    if not user_id:
        raise SystemExit("Could not pick a reviewerID from parquet.")

    rec_timings: dict[str, float] = {}
    recs = model.recommend_for_user(user_id, top_k=args.top_k, timings=rec_timings)

    print(f"User: {user_id}")
    print(f"Got {len(recs)} recommendations.")

    timings_out = {"build": build_timings, "recommend": rec_timings}
    print(json.dumps(timings_out, indent=2, sort_keys=True))

    if args.timings_json:
        Path(args.timings_json).write_text(json.dumps(timings_out, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

