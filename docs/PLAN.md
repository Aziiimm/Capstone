# Project Plan: GPU-Accelerated Hybrid Amazon Recommender

## 1. Project Overview & Scope

The objective is to implement a high-performance recommendation system using the Amazon Reviews 2023 dataset. We will compare a traditional CPU-based pipeline against a GPU-accelerated pipeline using NVIDIA RAPIDS (cuDF/cuML) and CUDA.

To enhance the system beyond simple ratings, we are implementing a Hybrid Logic that incorporates Sentiment Analysis of review text to refine recommendation accuracy and address the Cold-Start problem for new users.

### Key Performance Targets:

- **Speedup:** Aiming for a 10x–50x reduction in total end-to-end runtime on GPU.
- **Accuracy:** Maintaining or improving ranking quality compared to the ratings-only baseline.
- **Hardware Constraint:** Optimized for RTX 2080 Ti (11GB VRAM).

---

## 2. Detailed Technical Step-by-Step

### Phase 1: Data Acquisition & High-Efficiency Sampling

- **Target:** Create a manageable "Dev" dataset from the 750GB raw data.
- **Steps:**

1. Select a specific category (e.g., "Video Games").
2. Use Dask-cuDF for out-of-core loading to handle the initial JSON extraction.
3. **Pruning:** Retain only reviewerID, asin, rating, reviewText, and timestamp.
4. **Filtering:** Retain users with >5 reviews and items with >10 reviews to reduce matrix sparsity.
5. Save as Parquet for fast loading.

### Phase 2: Hybrid Feature Engineering (NLP + Ratings)

- **Target:** Generate a "Sentiment-Adjusted Rating" on the GPU.
- **Steps:**

1. **GPU Text Pre-processing:** Clean reviewText (lowercase, remove punctuation) using cuDF string kernels to avoid CPU bottlenecks.
2. **Sentiment Scoring:** Run a GPU-based sentiment analyzer (e.g., VADER or DistilBERT).
3. **Weighted Fusion:** Create hybrid*score = (0.7 * rating) + (0.3 \_ normalized_sentiment).
4. **Cold-Start Logic:** For new users, recommendations will be generated based purely on the sentiment analysis of their initial input.

### Phase 3: The Dual-Pipeline Implementation

- **Target:** Build both versions for head-to-head comparison.
- **GPU Pipeline (RAPIDS):**
- Implement ALS and KNN via cuML.

- **CPU Pipeline (Baseline):**
- Implement the same logic using Scikit-learn or Surprise.

- **Benchmarking:** Measure Data Transfer Overhead (Host-to-Device) alongside training and inference time to calculate the true Speedup Factor.

### Phase 4: Serving & Infrastructure

- **Target:** Deploy for real-time recommendation.
- **Inference Server:** Evaluate NVIDIA Triton vs. Flask for serving the GPU model.
- **Frontend:** React dashboard displaying recommended products and live "Speedup Factor".

---

## 3. Weekly Implementation Timeline

| Week      | Milestone          | Primary Tasks                                                | Deliverable         |
| --------- | ------------------ | ------------------------------------------------------------ | ------------------- |
| **1-2**   | **Data Pipeline**  | JSON extraction, filtering, and Parquet conversion.          | data_clean.py       |
| **3**     | **Text Cleaning**  | Implement cuDF string kernels for GPU-accelerated NLP.       | nlp_clean.ipynb     |
| **4**     | **Sentiment/CPU**  | Generate sentiment scores and build the CPU baseline.        | baseline_report.pdf |
| **5-6**   | **GPU ML Core**    | Implement cuML ALS; optimize for 11GB VRAM limit.            | gpu_trainer.py      |
| **7**     | **Hybrid Tuning**  | Optimize Rating/Sentiment weights and test Cold-Start logic. | Model Weights       |
| **8-10**  | **Integration**    | Connect Flask/Triton to React UI; finalize inference logic.  | Working API         |
| **11-12** | **Final Analysis** | Scaling tests, NDCG@K calculation, and Final Presentation.   | Final Project Repo  |

---

## 4. Success Metrics for the Final Report

1. **Speedup Factor:** Total runtime ratio including HtoD Transfer Time.
2. **NDCG@K:** Ranking quality (how well the model orders recommendations).
3. **Precision@K:** Accuracy of the top-K recommended items.
4. **Latency:** Milliseconds for a single user recommendation request.

---
