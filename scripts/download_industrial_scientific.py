"""
Download Industrial and Scientific review + meta from Hugging Face into dataset/.

Usage (from project root):
  pip install huggingface_hub
  python scripts/download_industrial_scientific.py

Output:
  dataset/Industrial_and_Scientific.jsonl
  dataset/meta_Industrial_and_Scientific.jsonl
"""
from __future__ import annotations

import os
import shutil

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
REVIEW_FILE = "raw/review_categories/Industrial_and_Scientific.jsonl"
META_FILE = "raw/meta_categories/meta_Industrial_and_Scientific.jsonl"


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "Install huggingface_hub first: pip install huggingface_hub"
        )

    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    dest_review = os.path.join(dataset_dir, "Industrial_and_Scientific.jsonl")
    dest_meta = os.path.join(dataset_dir, "meta_Industrial_and_Scientific.jsonl")

    if os.path.isfile(dest_review) and os.path.isfile(dest_meta):
        print("Files already present in dataset/. Skipping download.")
        print(f"  {dest_review}")
        print(f"  {dest_meta}")
        return

    print("Downloading from Hugging Face (McAuley-Lab/Amazon-Reviews-2023)...")
    print("Reviews (~2.35 GB)...")
    path_review = hf_hub_download(
        repo_id=REPO_ID,
        filename=REVIEW_FILE,
        repo_type="dataset",
    )
    shutil.copy2(path_review, dest_review)
    print(f"  -> {dest_review}")

    print("Meta (~1.13 GB)...")
    path_meta = hf_hub_download(
        repo_id=REPO_ID,
        filename=META_FILE,
        repo_type="dataset",
    )
    shutil.copy2(path_meta, dest_meta)
    print(f"  -> {dest_meta}")

    print("Done. Run the pipeline with:")
    print("  python -m data.run_pipeline --source dataset/Industrial_and_Scientific.jsonl --meta dataset/meta_Industrial_and_Scientific.jsonl --output output/sample.parquet --limit 50000")


if __name__ == "__main__":
    main()
