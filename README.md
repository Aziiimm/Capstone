# GPU-Accelerated Hybrid Amazon Recommender

Recommendation system on **Amazon Reviews 2023**, comparing a CPU baseline with a GPU pipeline (RAPIDS cuDF/cuML). Uses sentiment from review text.

---

## Quick start: CPU path (for the team)

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

Installs Dask, PyArrow, and Pandas for out-of-core processing.

### Step 2: Download your category data

Each category needs **two files** from the [UCSD datarepo](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023):

| Category                   | Review file                        | Meta file                               |
| -------------------------- | ---------------------------------- | --------------------------------------- |
| Electronics                | `Electronics.jsonl`                | `meta_Electronics.jsonl`                |
| Tools and Home Improvement | `Tools_and_Home_Improvement.jsonl` | `meta_Tools_and_Home_Improvement.jsonl` |
| Industrial and Scientific  | `Industrial_and_Scientific.jsonl`  | `meta_Industrial_and_Scientific.jsonl`  |

**Download paths:**

- Reviews: `review_categories/<Category>.jsonl.gz`
- Meta: `meta_categories/meta_<Category>.jsonl.gz`

### Step 3: Run the pipeline

**Test run first** (limit rows; ~5–10 min for 50k rows):

```bash
python -m data.run_pipeline --source dataset/<REVIEW_FILE> --meta dataset/<META_FILE> --output output/sample.parquet --limit 50000 --timings-file output/timings_sample.json
```

**Replace** `<REVIEW_FILE>` and `<META_FILE>` with your filenames, e.g.:

- `Tools_and_Home_Improvement.jsonl` and `meta_Tools_and_Home_Improvement.jsonl`
- `Electronics.jsonl` and `meta_Electronics.jsonl`
- `Industrial_and_Scientific.jsonl` and `meta_Industrial_and_Scientific.jsonl`

**Full run** (no limit; expect several hours for large categories):

```bash
python -m data.run_pipeline --source dataset/<REVIEW_FILE> --meta dataset/<META_FILE> --output output/dev_<CATEGORY>.parquet --timings-file output/timings_<CATEGORY>.json
```

### Step 4: Verify output

```bash
python -c "import pandas as pd; d = pd.read_parquet('output/sample.parquet'); print(d.shape); print(d.columns.tolist())"
```

Expected columns: `reviewerID`, `asin`, `rating`, `reviewText`, `timestamp`, `product_title`, `main_category`.

---

## Data schema (output Parquet)

| Column          | Description               |
| --------------- | ------------------------- |
| `reviewerID`    | User ID                   |
| `asin`          | Product ID                |
| `rating`        | Rating (1–5)              |
| `reviewText`    | Review body               |
| `timestamp`     | Unix time                 |
| `product_title` | Product name (from meta)  |
| `main_category` | Main category (from meta) |

Filtering: **users with ≥6 reviews** and **items with ≥11 reviews**.

---

## Performance

The pipeline prints wall-clock time per stage (load, filter, to_parquet). Use `--timings-file` to save timings for CPU vs GPU comparison:

```bash
python -m data.run_pipeline --source dataset/<REVIEW_FILE> --meta dataset/<META_FILE> --output output/dev_<CATEGORY>.parquet --timings-file output/timings_cpu.json
```

---

## Large data and GitHub

- Raw and output data are large. `.gitignore` excludes `dataset/` and `output/`.
- Do not commit data to Git. Use a shared drive or cloud storage for Parquet files.
- The pipeline is out-of-core (Dask) so large files are supported.

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Repo layout

```
Capstone/
  data/
    load.py         # Load review + meta, join, prune
    filter.py       # User ≥6, item ≥11 reviews
    to_parquet.py   # Write Parquet
    run_pipeline.py # CLI entry point
  dataset/          # Put your .jsonl or .jsonl.gz files here
  output/           # Parquet output
  tests/
  requirements.txt
```

---

## Citation

[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)