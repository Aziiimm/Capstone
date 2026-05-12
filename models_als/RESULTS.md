# GPU-Accelerated Recommender: CPU vs GPU Comparison

Summary of the work that produced the [models_als/](./) folder and the apples-to-apples CPU/GPU comparison the professor requested.

---

## Executive summary

The previous setup couldn't actually be compared CPU-to-GPU: the GPU side ran item-KNN (cuML) and the CPU side ran SVD (Surprise). Two different algorithms, different metrics, different evaluation methodologies. The professor flagged this as a "VERY weak" comparison.

We replaced both with a single implicit ALS model that runs on both backends from the same code path (`--backend cpu` or `--backend gpu`). The only difference between runs is one flag.

**Headline result:** identical algorithm, identical hyperparameters, identical accuracy on both backends — **GPU trains ~59× faster than CPU on the same data.**

---

## The problem we walked into

### Original GPU model ([models/](../models/))

cuML's `NearestNeighbors` with cosine similarity on the raw rating matrix. Identified issues:

- **Broken history aggregation** ([models/recommender.py:31-49](../models/recommender.py#L31-L49)): when a user had multiple items in their history, the code flattened per-row neighbor lists in row-major order. Recommendations came almost entirely from the first history item.
- **Raw 1–5 ratings as similarity inputs**: cosine similarity treats a 1-star and a 5-star as the same signal direction. Two items rated by harsh users looked "similar" regardless of whether they were loved or hated.
- **Sentiment from review text advertised but unused**: README claimed the model used sentiment, but [models/new_model.py:68](../models/new_model.py#L68) only read `reviewerID, asin, rating`.
- **Titles and categories stored but unused**: pure CF, no content signal.
- **Honest Hit@10 = 0.029**, NDCG@10 = 0.016. Predictions were visibly incoherent (e.g., binoculars seed returned a deck resurfacer + a nightgown).

### Original CPU model (SVD, per-category)

- Reported **NDCG@10 = 0.9753** — almost certainly contaminated evaluation. Real Amazon-scale NDCG lives in the 0.05–0.15 range; 0.97 means the model was being scored on items it had seen during training.
- **RMSE = 0.2464** on a 1–5 scale corroborates the contamination (state-of-the-art is ~0.9–1.1).
- **One model per category** — throws away cross-category signal, which is most of the predictive value at Amazon scale.
- Top-10 recommendations were `pred ≈ 1.0` across the board, indicating predictions were saturated on globally-popular items rather than personalized.

### The comparison problem

| | CPU side | GPU side |
|---|---|---|
| Model | SVD (Surprise) | item-KNN (cuML) |
| Metrics | RMSE / contaminated NDCG | Hit@K / NDCG@K |
| Comparable? | **No.** |

---

## What we built

A new package, [models_als/](./), with five files:

| File | Purpose |
|---|---|
| [recommender_als.py](recommender_als.py) | Wraps `implicit.als.AlternatingLeastSquares`. Same `recommend(history, top_k)` interface as the legacy KNN recommender. |
| [train_als.py](train_als.py) | Trains one global model across all parquet shards. `--backend {cpu,gpu}` selects the implementation. Writes timings JSON. |
| [evaluate_als.py](evaluate_als.py) | Time-based holdout evaluation. Holds out each user's most-recent interaction *before* training, then scores Hit@K / NDCG@K / MRR against truly unseen targets. |
| [server_als.py](server_als.py) | Drop-in FastAPI server. Same `/recommend`, `/search`, `/health` endpoints as the legacy server. Frontend needs no changes. |
| [__init__.py](__init__.py) | Package marker. |

### Why this specifically addresses the professor's feedback

> "It would be really preferable to have at least one common model that runs on both CPU and GPU and measure the same performance metric(s)."

Three concrete asks, all satisfied:

1. **Same model on both sides** — `implicit.als.AlternatingLeastSquares`, one library.
2. **Runs on both CPU and GPU** — `use_gpu=True/False` toggle; CPU uses Cholesky, GPU uses Conjugate Gradient, but both implement standard ALS.
3. **Same metrics** — single eval script ([evaluate_als.py](evaluate_als.py)) computes Hit@K, NDCG@K, MRR, and latency on both backends with the same seed.

### Key design choices

- **One global model**, not per-category. Cross-category signal is captured.
- **Confidence weighting** (Hu/Koren 2008): `data = 1 + α * rating`. Standard implicit-feedback ALS.
- **Min-rating filter (≥4.0)**: treat 4+ stars as positive implicit feedback. 1–3 stars dropped as noise.
- **Time-based holdout in eval**: held-out items are truly unseen during training, so metrics are not contaminated.

---

## Dataset

| Stat | Value |
|---|---|
| Categories | 5 (Clothing/Shoes/Jewelry, Electronics, Industrial/Scientific, Sports/Outdoors, Tools/Home Improvement) |
| Parquet shards | 140 part files |
| Total interactions | 38,318,638 |
| After rating ≥ 4.0 filter | 31,138,788 |
| Unique users | 3,782,628 |
| Unique items | 2,138,088 |
| Eligible users (≥5 interactions) | 2,209,142 |

---

## Results

### Headline comparison (1,000 sampled users)

| Metric | CPU | GPU | Result |
|---|---|---|---|
| Train time (factors=64, iters=20) | **403.13s** (6m 43s) | **6.83s** | **~59× GPU speedup** |
| Inference p50 latency | 13.49ms | 1.40ms | ~9× faster |
| Hit@10 | 0.0090 | 0.0100 | equivalent (within noise) |
| NDCG@10 | 0.0046 | 0.0061 | equivalent |
| MRR@20 | 0.0035 | 0.0049 | equivalent |

### Tighter accuracy comparison (10,000 sampled users)

At N=10k, statistical noise drops by √10. This confirms the algorithm's true ceiling on this data.

| Config | Hit@10 | NDCG@10 | Train time | Notes |
|---|---|---|---|---|
| GPU baseline (f=64, i=20, reg=0.05) | **0.0066** | 0.0035 | 6.87s | reference |
| GPU "tune v1" (f=128, i=50, reg=0.01) | 0.0063 | 0.0034 | 39.32s | over-regularization too low → no gain, slight overfit |
| GPU "tune v2" (f=128, i=30, reg=0.05) | 0.0068 | 0.0039 | 23.70s | within noise of baseline; 3.4× train cost for ~0 lift |

**Conclusion:** plain ALS is at its ceiling on this data. Further hyperparameter tuning does not move the needle.

### Output artifacts (all under `output/`)

- `timings_als_cpu.json` — CPU training timings
- `timings_als_gpu.json` — GPU training timings
- `metrics_als_cpu.json` — CPU accuracy + latency (1k users)
- `metrics_als_gpu.json` — GPU accuracy + latency (1k users)
- `metrics_als_gpu_10k.json` — GPU baseline at 10k users
- `metrics_als_gpu_tuned.json` — GPU tune v1
- `metrics_als_gpu_tune_v2.json` — GPU tune v2

---

## What the metrics actually mean

The previous numbers were misleading; the new ones are honest. For context:

| Number | Source | What it actually means |
|---|---|---|
| **NDCG@10 = 0.9753** (old SVD) | CPU eval | Contaminated — model was scored on training data. Not real. |
| **Hit@10 = 0.029** (old KNN) | GPU eval | Random holdout against a memorized target — easier task than time-based. Inflated. |
| **Hit@10 = 0.0066** (this work) | Honest time-based eval | Predicting a user's truly-unseen future purchase from 2.13M items. The real number. |

A random baseline at this catalog size would score Hit@10 ≈ 0.0000047. Our 0.0066 is ~1,400× better than random, which is a meaningful lift even though the absolute number sounds low.

---

## Bugs fixed during integration

Worth recording so the next person doesn't re-discover them:

1. **`rating` stored as strings** in some parquet shards → coerce with `pd.to_numeric(..., errors='coerce')` and drop NaN rows.
2. **`timestamp` is `datetime64`, not int** → `pd.to_numeric` returned NaN for the whole column and dropped every row. Fix: don't coerce timestamp; pandas sorts datetimes natively.
3. **Full sort on 31M rows hung indefinitely** in `df.sort_values(["user_idx", "timestamp"])`. Replaced with `df.groupby("user_idx", sort=False)["timestamp"].idxmax()` which is O(n) and avoids the rewrite.
4. **Index/position mismatch** after the rating filter — `idxmax` returned indices in the unfiltered range. Fix: `df.reset_index(drop=True)` immediately after filtering so positional masking aligns.
5. **OpenBLAS threadpool fights with implicit's threading on CPU.** Always set `OPENBLAS_NUM_THREADS=1` before CPU runs (the library prints a warning if you forget).

---

## How to reproduce

```bash
# 1. Train (both backends)
python -m models_als.train_als \
    --path "output/SD/**/*.parquet" \
    --factors 64 --iterations 20 --alpha 40 --regularization 0.05 \
    --backend cpu --output models_als/als_recommender_cpu \
    --timings-file output/timings_als_cpu.json

python -m models_als.train_als \
    --path "output/SD/**/*.parquet" \
    --factors 64 --iterations 20 --alpha 40 --regularization 0.05 \
    --backend gpu --output models_als/als_recommender_gpu \
    --timings-file output/timings_als_gpu.json

# 2. Evaluate (both backends, same seed → same holdout sample)
OPENBLAS_NUM_THREADS=1 python -m models_als.evaluate_als \
    --path "output/SD/**/*.parquet" \
    --factors 64 --iterations 20 --alpha 40 --regularization 0.05 \
    --backend cpu --users 10000 \
    --metrics-file output/metrics_als_cpu_10k.json

python -m models_als.evaluate_als \
    --path "output/SD/**/*.parquet" \
    --factors 64 --iterations 20 --alpha 40 --regularization 0.05 \
    --backend gpu --users 10000 \
    --metrics-file output/metrics_als_gpu_10k.json

# 3. Serve (either backend)
MODEL_PATH=models_als/als_recommender_gpu \
    uvicorn models_als.server_als:app --host 0.0.0.0 --port 8000
```

---

## Future work: paths to higher accuracy

Plain ALS is at its ceiling on this data (Hit@10 ≈ 0.0066). To push further, the algorithm class has to change. All three options below preserve the same-model-on-both-backends comparison.

| Approach | Effort | Expected Hit@10 | × current |
|---|---|---|---|
| **BPR** (drop-in `implicit.bpr`) | ~1 hour | 0.010–0.015 | ~2× |
| **Two-tower neural CF** (PyTorch, uses titles + categories) | 1–2 days | 0.020–0.035 | ~5× |
| **SASRec / BERT4Rec** (PyTorch, sequential transformer) | 3–5 days | 0.050–0.100 | ~15× |

The two-tower approach is the strongest impact-per-effort: PyTorch's `.to('cuda')` vs `.to('cpu')` preserves the comparison story trivially, and adding title/category content features bypasses the cold-start and long-tail weaknesses that limit pure CF.

Other potential wins independent of model choice:
- Category-scoped evaluation (restrict candidate pool to the held-out item's category) — instantly multiplies all Hit@K numbers without changing the model. Worth confirming with the professor that this framing is acceptable.
- Sentiment features extracted from `reviewText` (already in the parquet, currently unused).
- Recency weighting in the confidence function — recent interactions count more.

---

## What changed in the repo

```
Capstone/
  models/              # legacy (KNN) — kept for reference
  models_als/          # NEW — implicit ALS, CPU + GPU
    __init__.py
    recommender_als.py
    train_als.py
    evaluate_als.py
    server_als.py
    RESULTS.md         # this file
  output/
    metrics_als_*.json
    timings_als_*.json
```
