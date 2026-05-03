import argparse
import json
import logging
import time
import os
import gc
from pathlib import Path

import cudf
import rmm
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

rmm.reinitialize(
    pool_allocator=True,
    initial_pool_size=int(10e9),  # Freed up 2GB for the model in VRAM
    managed_memory=True
)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_WORDS = 120   # Pre-truncate on GPU BEFORE tokenizing — big CPU time saver
BATCH_SIZE = 512  # Safe for DistilBERT fp16 on 24GB; tune up if you have headroom
NUM_WORKERS = 4   # Parallel CPU tokenizer workers to keep GPU fed

def clean_text_gpu(df: cudf.DataFrame, text_col: str) -> cudf.Series:
    log.info("Cleaning + pre-truncating text on GPU...")
    t0 = time.perf_counter()
    cleaned = (
        df[text_col].str.lower()
                    .str.replace(r"[^\w\s]", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                    .fillna("neutral review")
    )

    # Character-based truncation: ~120 words * ~6 chars/word = 720 chars.
    # str.slice() is a single GPU kernel — no list column construction,
    # no cuDF version concerns. The tokenizer hard-caps at max_length=128
    # anyway, so this just cuts the CPU tokenizer's workload dramatically.
    truncated = cleaned.str.slice(0, 750)

    log.info("  GPU clean+truncate: %.2fs", time.perf_counter() - t0)
    return truncated

class ReviewDataset(Dataset):
    """
    Wraps a plain Python list for torch DataLoader.
    Tokenization happens inside __getitem__ so DataLoader workers
    can parallelize it across CPU cores — this is what keeps the GPU fed.
    """
    def __init__(self, texts: list, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # This runs in a DataLoader worker process (not the main process),
        # so multiple cores tokenize concurrently.
        enc = self.tokenizer(
            self.texts[idx],
            max_length=128,       # Hard cap — pre-truncation keeps most under this
            truncation=True,
            padding="max_length", # Needed for stacking into a batch tensor
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def score_sentiment_transformer(df: cudf.DataFrame, text_col: str) -> list:
    log.info("Loading tokenizer + model...")
    # Use the fast (Rust-backed) tokenizer — it's 3-5x faster than the Python one
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to("cuda").eval()

    # Pull texts to CPU once as a list — avoid repeated GPU-CPU transfers in the loop
    log.info("Transferring texts to CPU for DataLoader...")
    texts = df[text_col].to_arrow().to_pylist()

    dataset = ReviewDataset(texts, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,   # Parallel tokenization across CPU cores
        pin_memory=True,           # Faster CPU→GPU transfer via pinned memory
        prefetch_factor=2,         # Each worker pre-fetches 2 batches ahead
        persistent_workers=True,   # Don't respawn workers between batches
    )

    log.info("Running inference (batch_size=%d, workers=%d)...", BATCH_SIZE, NUM_WORKERS)
    t0 = time.perf_counter()
    all_scores = []
    pos_idx = model.config.label2id.get("POSITIVE", 1)

    with torch.inference_mode():
        for batch in loader:
            # Non-blocking transfers overlap with the next DataLoader prefetch
            input_ids      = batch["input_ids"].to("cuda", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs  = torch.softmax(logits, dim=-1)

            pos_probs = probs[:, pos_idx].cpu().numpy()
            # Map to [-1, 1]: positive → score, negative → -score
            neg_probs = 1.0 - pos_probs
            scores = np.where(pos_probs >= neg_probs, pos_probs, -neg_probs)
            all_scores.extend(scores.tolist())

    elapsed = time.perf_counter() - t0
    log.info("Inference done: %.2fs  (%.0f rows/s)", elapsed, len(texts) / elapsed)
    return all_scores


def build_hybrid_score_gpu(df: cudf.DataFrame) -> cudf.Series:
    rating_norm    = (df["rating"].astype("float32") - 1.0) / 4.0
    sentiment_norm = (df["sentiment_compound"] + 1.0) / 2.0
    return (0.7 * rating_norm) + (0.3 * sentiment_norm)


def run_pipeline(input_path: str, output_path: str) -> dict:
    timings = {}

    log.info("Loading %s via cuDF...", input_path)
    t0 = time.perf_counter()
    df = cudf.read_parquet(input_path)
    timings["load_s"] = round(time.perf_counter() - t0, 3)

    df["clean_text"] = clean_text_gpu(df, "reviewText")

    t0 = time.perf_counter()
    df["sentiment_compound"] = score_sentiment_transformer(df, "clean_text")
    timings["sentiment_s"] = round(time.perf_counter() - t0, 3)

    df["hybrid_score"] = build_hybrid_score_gpu(df)

    log.info("Writing enriched Parquet...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.drop(columns=["clean_text"]).to_parquet(output_path)

    del df
    gc.collect()
    torch.cuda.empty_cache()

    return timings


def main():
    parser = argparse.ArgumentParser(description="Optimized GPU Sentiment Pipeline")
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--timings-file", default=None)
    args = parser.parse_args()

    timings = run_pipeline(args.input, args.output)

    if args.timings_file:
        with open(args.timings_file, "w") as f:
            json.dump({"backend": "titan_rtx_optimized", **timings}, f, indent=2)


if __name__ == "__main__":
    main()