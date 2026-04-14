from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CategoryFiles:
    review_relpath: str
    meta_relpath: str


def _pick_one(label: str, matches: Sequence[str]) -> str:
    if not matches:
        raise ValueError(f"No {label} file match found in dataset repo.")
    if len(matches) > 1:
        # Prefer shorter paths (usually the canonical one) to avoid picking benchmark artifacts.
        matches = sorted(matches, key=len)
    return matches[0]


def _find_category_files_in_repo(files: Iterable[str], category: str) -> CategoryFiles:
    """
    Find the correct review/meta file paths for a category by searching the dataset file list.

    The Amazon-Reviews-2023 dataset layout has changed over time (e.g. raw/* prefixes, .jsonl vs .jsonl.gz).
    We therefore discover the actual filenames instead of hardcoding.
    """
    cat = category.strip()
    if not cat:
        raise ValueError("Category must be a non-empty string.")

    file_list = list(files)

    # Look for review files like ".../review_categories/<Category>.jsonl(.gz)"
    review_candidates = [
        f
        for f in file_list
        if f.endswith((".jsonl", ".jsonl.gz"))
        and "/review_categories/" in f.replace("\\", "/")
        and f.split("/")[-1].startswith(cat + ".")
    ]
    review_relpath = _pick_one("review", review_candidates)

    # Look for meta files like ".../meta_categories/meta_<Category>.jsonl(.gz)"
    meta_name_prefix = f"meta_{cat}."
    meta_candidates = [
        f
        for f in file_list
        if f.endswith((".jsonl", ".jsonl.gz"))
        and "/meta_categories/" in f.replace("\\", "/")
        and f.split("/")[-1].startswith(meta_name_prefix)
    ]
    meta_relpath = _pick_one("meta", meta_candidates)

    return CategoryFiles(review_relpath=review_relpath, meta_relpath=meta_relpath)


def download_category(
    category: str,
    out_dir: str = "dataset",
    *,
    repo_id: str = "McAuley-Lab/Amazon-Reviews-2023",
) -> tuple[str, str]:
    """
    Download one category's review + meta .jsonl.gz files into out_dir.

    Returns
    -------
    (review_path, meta_path) as local filesystem paths.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for downloading. "
            "Install it with: `pip install huggingface_hub`"
        ) from e

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    files = _find_category_files_in_repo(repo_files, category)
    out_dir_abs = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir_abs, exist_ok=True)

    review_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=files.review_relpath,
        local_dir=out_dir_abs,
    )
    meta_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=files.meta_relpath,
        local_dir=out_dir_abs,
    )
    return review_path, meta_path


def main() -> None:
    p = argparse.ArgumentParser(description="Download Amazon Reviews 2023 category files.")
    p.add_argument(
        "--category",
        required=True,
        help="Category folder name, e.g. Tools_and_Home_Improvement, Electronics, Industrial_and_Scientific.",
    )
    p.add_argument(
        "--out-dir",
        default="dataset",
        help="Where to place the downloaded .jsonl.gz files (default: dataset/).",
    )
    args = p.parse_args()

    review_path, meta_path = download_category(args.category, out_dir=args.out_dir)
    print("Downloaded:")
    print(" reviews:", review_path)
    print(" meta:   ", meta_path)


if __name__ == "__main__":
    main()

