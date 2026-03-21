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
    run_pipeline.py         # Single-category CLI entry point
    run_multi_categories.py # Multi-category orchestrator (server runs)
  dataset/          # Put your .jsonl or .jsonl.gz files here
  output/           # Parquet output
  logs/             # Multi-category run logs
  tests/
  requirements.txt
```

---

## Multi-category pipeline (CPU vs GPU, server runs)

For running several Amazon Review categories **sequentially** on the server (with automatic cleanup), use `data/run_multi_categories.py`.

- **CPU baseline (your part)**: **omit** `--gpu`.
- **GPU comparison (CUDA/RAPIDS)**: **include** `--gpu`.

### 1. Configure categories

Categories are defined in `categories.json` at the repo root. Example (using plain `.jsonl` files):

```json
{
  "categories": [
    {
      "name": "Electronics",
      "review_file": "Electronics.jsonl",
      "meta_file": "meta_Electronics.jsonl"
    },
    {
      "name": "Tools_and_Home_Improvement",
      "review_file": "Tools_and_Home_Improvement.jsonl",
      "meta_file": "meta_Tools_and_Home_Improvement.jsonl"
    },
    {
      "name": "Industrial_and_Scientific",
      "review_file": "Industrial_and_Scientific.jsonl",
      "meta_file": "meta_Industrial_and_Scientific.jsonl"
    }
  ]
}
```

- **name**: Logical name for logging and output filenames.
- **review_file**: Review JSONL/JSONL.GZ file in `dataset/` (e.g. `Electronics.jsonl` or `Electronics.jsonl.gz`).
- **meta_file**: Meta JSONL/JSONL.GZ file in `dataset/` (e.g. `meta_Electronics.jsonl` or `meta_Electronics.jsonl.gz`).

All files listed here should be **pre-downloaded** into the `dataset/` directory on the GPU server.

### 2. Run the multi-category script

From the repo root on the server:

- **CPU baseline run (no GPU, your part):**

  ```bash
  python -m data.run_multi_categories --config categories.json --output-dir output
  ```

- **GPU run (CUDA/RAPIDS comparison):**

  ```bash
  python -m data.run_multi_categories --config categories.json --gpu --output-dir output
  ```

Useful flags:

- `--gpu`: Use the GPU pipeline (Dask-cuDF) if available.
- `--limit N`: Limit rows per category (for quick tests).
- `--blocksize SIZE`: Dask read block size (default `64MB`).
- `--output-dir DIR`: Base directory for Parquet and per-category timings (default `output/`).
- `--summary-timings PATH`: Optional path for an aggregated summary JSON (default `output/timings_multi_summary.json`).
- `--continue-on-error`: Continue with remaining categories even if one fails.
- `--no-delete-inputs`: Do **not** delete the input `.jsonl` files after a successful category (useful for debugging).

### 3. Outputs, timings, and logging

For each category `<CATEGORY>` (derived from `name` in `categories.json` with spaces replaced by underscores), the script writes:

- Parquet: `output/dev_<CATEGORY>.parquet`
- Per-category timings: `output/timings_<CATEGORY>.json`

It also writes an overall summary file (by default):

- `output/timings_multi_summary.json`

This JSON includes:

- Backend (`cpu`/`gpu`)
- Absolute paths for `config` and `output_dir`
- Flags (`delete_inputs`, `continue_on_error`)
- A `categories` list with per-category status, timings, and paths

### 4. Automatic cleanup of JSONL inputs

By default, after a category finishes **successfully**, the script will:

- Delete that category’s review JSONL/JSONL.GZ file from `dataset/`
- Delete that category’s meta JSONL/JSONL.GZ file from `dataset/`

This helps keep storage usage low on the professor’s GPU server.

Safety details:

- Only files under the `dataset/` directory are eligible for deletion.
- Files are deleted **after** the Parquet file and per-category timings JSON have been written.
- If a category fails, its input files are **not** deleted.
- Use `--no-delete-inputs` to disable this behavior (e.g., for debugging or when you do not want cleanup).

### 5. Logs vs timings

In addition to timings JSON files, the multi-category run writes detailed logs to:

- `logs/multi_run_<YYYYMMDD_HHMMSS>.log`

Logging includes:

- Start/end of the overall multi-category run
- Start/end of each category
- Input/output paths used
- Per-stage timings (load, filter, to_parquet)
- Exceptions and stack traces on failure
- File deletions and any deletion errors

Logs are written to **both**:

- Console/stdout (so you can tail the run interactively)
- The log file in `logs/` (for later inspection or sharing)

---

## Citation

[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
