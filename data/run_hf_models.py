"""
Run Hugging Face model tests on pipeline output and write enriched data.

This script is designed as a safe post-processing step:
- It reads a Parquet dataset produced by `data.run_pipeline`.
- It benchmarks one or more sentiment and embedding models.
- It enriches the sampled rows with features from one selected model.
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd


DEFAULT_SENTIMENT_MODELS = [
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
]

DEFAULT_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
]


def _normalize_text_col(series: pd.Series) -> list[str]:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .tolist()
    )


def benchmark_sentiment(
    texts: list[str], model_id: str, batch_size: int = 16, device: int = -1
) -> dict:
    from transformers import pipeline

    clf = pipeline(
        "sentiment-analysis",
        model=model_id,
        device=device,
        truncation=True,
    )
    start = time.perf_counter()
    outputs = clf(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - start
    rows_per_sec = len(texts) / elapsed if elapsed > 0 else 0.0
    return {
        "model": model_id,
        "task": "sentiment",
        "rows": len(texts),
        "elapsed_s": elapsed,
        "rows_per_sec": rows_per_sec,
        "sample": outputs[0] if outputs else {},
    }


def enrich_with_sentiment(
    df: pd.DataFrame,
    text_col: str,
    model_id: str,
    batch_size: int = 16,
    device: int = -1,
) -> pd.DataFrame:
    from transformers import pipeline

    texts = _normalize_text_col(df[text_col])
    clf = pipeline(
        "sentiment-analysis",
        model=model_id,
        device=device,
        truncation=True,
    )
    preds = clf(texts, batch_size=batch_size)
    out = df.copy()
    out["hf_sentiment_model"] = model_id
    out["hf_sentiment_label"] = [p.get("label") for p in preds]
    out["hf_sentiment_score"] = [float(p.get("score", 0.0)) for p in preds]
    return out


def benchmark_embedding(texts: list[str], model_id: str, batch_size: int = 32) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    start = time.perf_counter()
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    elapsed = time.perf_counter() - start
    rows_per_sec = len(texts) / elapsed if elapsed > 0 else 0.0
    return {
        "model": model_id,
        "task": "embedding",
        "rows": len(texts),
        "embedding_dim": int(vecs.shape[1]) if len(vecs.shape) == 2 else 0,
        "elapsed_s": elapsed,
        "rows_per_sec": rows_per_sec,
    }


def enrich_with_embeddings(
    df: pd.DataFrame, text_col: str, model_id: str, batch_size: int = 32
) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    texts = _normalize_text_col(df[text_col])
    model = SentenceTransformer(model_id)
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    out = df.copy()
    out["hf_embedding_model"] = model_id
    out["hf_embedding_dim"] = int(vecs.shape[1])
    # Store as compact JSON strings to keep parquet schema simple.
    out["hf_embedding_vector"] = [json.dumps(row.tolist()) for row in vecs]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Hugging Face models and enrich sample data."
    )
    parser.add_argument("--input", required=True, help="Input Parquet path")
    parser.add_argument("--output", required=True, help="Output Parquet path")
    parser.add_argument(
        "--text-col", default="reviewText", help="Text column name (default reviewText)"
    )
    parser.add_argument(
        "--rows", type=int, default=1000, help="Rows to sample for model tests"
    )
    parser.add_argument(
        "--sentiment-models",
        nargs="*",
        default=DEFAULT_SENTIMENT_MODELS,
        help="Sentiment model IDs to benchmark",
    )
    parser.add_argument(
        "--embedding-models",
        nargs="*",
        default=DEFAULT_EMBEDDING_MODELS,
        help="Embedding model IDs to benchmark",
    )
    parser.add_argument(
        "--use-sentiment-model",
        default=DEFAULT_SENTIMENT_MODELS[0],
        help="Sentiment model used for output enrichment",
    )
    parser.add_argument(
        "--use-embedding-model",
        default=DEFAULT_EMBEDDING_MODELS[0],
        help="Embedding model used for output enrichment",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for model inference"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Transformers device (-1 CPU, 0 first GPU)",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write benchmark JSON report",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if args.text_col not in df.columns:
        raise ValueError(
            f"Text column '{args.text_col}' not found. Available columns: {list(df.columns)}"
        )
    sample = df.head(args.rows).copy()
    texts = _normalize_text_col(sample[args.text_col])

    report: dict[str, list[dict]] = {"sentiment": [], "embedding": []}

    print(f"Loaded {len(df)} rows; testing with first {len(sample)} rows.")

    for model_id in args.sentiment_models:
        print(f"Benchmarking sentiment model: {model_id}")
        report["sentiment"].append(
            benchmark_sentiment(
                texts=texts, model_id=model_id, batch_size=args.batch_size, device=args.device
            )
        )

    for model_id in args.embedding_models:
        print(f"Benchmarking embedding model: {model_id}")
        report["embedding"].append(
            benchmark_embedding(texts=texts, model_id=model_id, batch_size=args.batch_size)
        )

    print(f"Enriching rows with sentiment model: {args.use_sentiment_model}")
    enriched = enrich_with_sentiment(
        sample,
        text_col=args.text_col,
        model_id=args.use_sentiment_model,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"Enriching rows with embedding model: {args.use_embedding_model}")
    enriched = enrich_with_embeddings(
        enriched,
        text_col=args.text_col,
        model_id=args.use_embedding_model,
        batch_size=max(args.batch_size, 8),
    )

    enriched.to_parquet(args.output, index=False)
    print(f"Enriched sample written to {args.output}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Benchmark report written to {args.report_json}")

    # Print compact report summary to terminal.
    for task in ("sentiment", "embedding"):
        print(f"\n{task.upper()} RESULTS")
        for row in report[task]:
            rps = row.get("rows_per_sec", 0.0)
            model = row.get("model", "unknown")
            extra = ""
            if task == "embedding":
                extra = f", dim={row.get('embedding_dim', 0)}"
            print(f"- {model}: {rps:.2f} rows/s{extra}")


if __name__ == "__main__":
    main()
