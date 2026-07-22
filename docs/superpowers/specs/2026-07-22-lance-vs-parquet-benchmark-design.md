# Lance vs Parquet Benchmark — Design Spec

| | |
|---|---|
| Status | Approved |
| Date | 2026-07-22 |
| Goal | Internal PoC to demonstrate Lance format advantages over Parquet for AI model training data loading via S3-compatible storage (MinIO) |

---

## 1. Overview

A self-contained benchmark project that generates realistic GAIA fuel-delivery feature data, stores it in both Lance and Parquet formats on MinIO, runs standardized access-pattern and model-training benchmarks, and produces an HTML report comparing performance metrics.

The audience is the internal ML/data engineering team. The goal is to produce hard evidence that Lance is worth adopting for the production GAIA lakehouse training pipeline.

---

## 2. Project Structure

```
lance-sample/
├── docker-compose.yml          # MinIO single container
├── pyproject.toml              # uv-managed deps
├── src/
│   ├── __init__.py
│   ├── config.py               # MinIO credentials, bucket names, data scale constants
│   ├── generate.py             # Synthetic GAIA data generator
│   ├── storage.py              # Upload/manage files on MinIO (both formats)
│   ├── benchmark.py            # Core benchmark runner (all access patterns + training)
│   ├── metrics.py              # Timing, memory, CPU, S3 call tracking utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── decision_xgb.py    # XGBoost Decision Model (classify: deliver today?)
│   │   └── volume_mlp.py      # PyTorch MLP Volume Model (regress: how much?)
│   └── report.py              # HTML report generation with Plotly charts
├── run.sh                      # One-liner entry point
├── ref/                        # Reference docs (existing)
└── docs/                       # Design docs
```

### How to run

```bash
git clone <repo> && cd lance-sample
docker compose up -d            # starts MinIO
uv sync                         # install Python deps
uv run python -m src.benchmark  # generates data, runs benchmarks, outputs report
open report.html                # view results
```

Three commands from zero to results.

---

## 3. Infrastructure

### MinIO

- Single Docker container
- Image: `minio/minio:RELEASE.2024-06-13T22-53-53Z` (pinned)
- Ports: 9000 (S3 API), 9001 (web console)
- Single bucket: `benchmark`
- Paths: `s3://benchmark/parquet/decision/`, `s3://benchmark/parquet/volume/`, `s3://benchmark/lance/decision/`, `s3://benchmark/lance/volume/`
- Credentials: static access key/secret in docker-compose (demo only, not production)

### docker-compose.yml

```yaml
services:
  minio:
    image: minio/minio:RELEASE.2024-06-13T22-53-53Z
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  minio_data:
```

---

## 4. Data Generation

### Dataset 1: Decision Model Features (wide table)

Mirrors the Gold `gold_decision_model_features` table from the GAIA spec.

- **Rows**: ~547,500 (500 sites × 3 tanks × 365 days)
- **Columns** (30+):

| Column | Type | Description |
|--------|------|-------------|
| `site_code` | string | e.g. "I001"–"I500" |
| `tank_id` | string | e.g. "T01"–"T03" |
| `feature_date` | date | 2025-07-22 to 2026-07-21 (365 days) |
| `product_code` | string | one of: "5000018" (HSD), "5000011" (ULG91), "5000012" (ULG95) |
| `region_code` | string | one of: "NORTH", "SOUTH", "EAST", "WEST", "CENTRAL" |
| `tank_capacity` | float64 | 15000–40000 |
| `open_inventory` | float64 | 0.1–0.9 × capacity |
| `avg_sale_7d` | float64 | 500–5000 |
| `avg_sale_30d` | float64 | 500–5000 |
| `overdue_ratio` | float64 | 0.0–3.0 |
| `inv_days_cover` | float64 | 0.0–15.0 |
| `dow_hist_rate` | float64 | 0.0–1.0 |
| `hist_rate` | float64 | 0.0–1.0 |
| `delivery_sum_28d` | float64 | 0–200000 |
| `proj_end_fill_ratio` | float64 | 0.0–1.0 |
| `group_overdue_pct` | float64 | 0.0–1.0 |
| `group_cover_pct` | float64 | 0.0–1.0 |
| `last_delivery_days_ago` | int32 | 0–30 |
| `intransit_volume` | float64 | 0–30000 |
| `usage_day_at_approval` | float64 | 0.5–10.0 |
| `current_inventory_at_approval` | float64 | derived |
| `day_of_week` | int8 | 0–6 |
| `day_of_month` | int8 | 1–31 |
| `month` | int8 | 1–12 |
| `is_weekend` | bool | |
| `is_holiday` | bool | |
| `delivery_volume_lag1` | float64 | previous day's delivery |
| `delivery_volume_lag7` | float64 | 7 days ago delivery |
| `rolling_delivery_count_7d` | int32 | |
| `rolling_delivery_count_30d` | int32 | |
| `pair_deliv_mean` | float64 | site-product pair delivery average |
| `yoy_deliv` | float64 | year-over-year delivery volume |
| `delivery_occurred` | int8 | **target** — binary 0/1 |

**Data realism**: Features have correlated distributions. Higher `overdue_ratio` correlates with `delivery_occurred=1`. Seasonal patterns in `dow_hist_rate`. Not random noise — enough structure for a model to learn, making the training benchmark meaningful.

### Dataset 2: Hourly Events (narrow, time-series)

Mirrors the Gold ATG hourly table used by the Volume Model.

- **Rows**: ~13,140,000 (500 sites × 3 tanks × 365 days × 24 hours)
- **Columns** (6):

| Column | Type | Description |
|--------|------|-------------|
| `reading_date` | date | 2025-07-22 to 2026-07-21 |
| `reading_hour` | int8 | 0–23 |
| `site_code` | string | "I001"–"I500" |
| `tank_id` | string | "T01"–"T03" |
| `atg_start` | float64 | tank level at hour start |
| `atg_diff` | float64 | consumption during hour (target proxy) |

**Data realism**: `atg_diff` follows a daily consumption pattern (peak hours 7–9, 17–19), with noise. `atg_start` decreases through the day then jumps on delivery events.

### Generation process

1. Generate data in-memory using NumPy with `seed=42`
2. Create a single PyArrow Table for each dataset
3. Write Parquet directly to MinIO: `pq.write_to_dataset(table, 's3://benchmark/parquet/decision/', partition_cols=['feature_date'], filesystem=s3fs)`
4. Write Lance directly to MinIO: `lance.write_dataset(table, 's3://benchmark/lance/decision/', storage_options={...})` — Lance writes as a flat dataset (no hive-style partitioning), relying on its internal fragment structure for efficient access
5. Repeat for the Volume (hourly events) dataset
6. Record file sizes for the report

---

## 5. Benchmark Tests

All benchmarks: 1 warmup run (discarded) + 3 measured repetitions. `gc.collect()` between runs.

### 5.1 Access Pattern Benchmarks

#### Benchmark 1: Random Row Access

- **Dataset**: Decision Model features (547K rows)
- **Operation**: Randomly sample 10,000 row indices, read those exact rows
- **Lance**: `dataset.take(indices)` — O(1) per row via index
- **Parquet**: `pq.read_table(filters=...)` or `dataset.take(indices)` via PyArrow — must scan row groups
- **Metrics**: wall-clock time, peak memory, first-row latency, S3 GET call count, CPU %
- **Expected outcome**: Lance significantly faster (10-50x) due to native random access

#### Benchmark 2: Sequential Full-Table Scan

- **Dataset**: Decision Model features (547K rows)
- **Operation**: Read entire table into Arrow Table
- **Lance**: `lance.dataset(uri).to_table()`
- **Parquet**: `pq.read_table(uri)`
- **Metrics**: wall-clock time, throughput (rows/sec, MB/sec), peak memory, CPU %
- **Expected outcome**: Similar performance — both formats are efficient at sequential reads. Parquet may be slightly faster due to maturity of implementation.

#### Benchmark 3: Column Subset Selection

- **Dataset**: Decision Model features (547K rows)
- **Operation**: Read only 8 of 30+ columns: `overdue_ratio`, `inv_days_cover`, `dow_hist_rate`, `hist_rate`, `avg_sale_7d`, `open_inventory`, `tank_capacity`, `delivery_occurred`
- **Lance**: `dataset.to_table(columns=[...])`
- **Parquet**: `pq.read_table(columns=[...])`
- **Metrics**: wall-clock time, throughput, peak memory, S3 GET call count, CPU %
- **Expected outcome**: Both should be efficient (columnar). Lance may have slight edge due to column metadata locality over S3.

### 5.2 Model Training Benchmarks

#### Benchmark 4: XGBoost Decision Model (end-to-end)

- **Dataset**: Decision Model features
- **Steps timed individually**:
  1. `data_load` — read from MinIO into Arrow Table
  2. `to_dmatrix` — convert Arrow Table → numpy → `xgb.DMatrix`
  3. `train` — `xgb.train(params, dtrain, num_boost_round=100)`
  4. `total` — end-to-end
- **XGBoost params** (fixed, not tuned):
  ```python
  {
      "objective": "binary:logistic",
      "max_depth": 6,
      "eta": 0.1,
      "eval_metric": "logloss",
      "nthread": 4,
  }
  ```
- **Expected outcome**: Difference visible in `data_load` step. `train` step identical (same DMatrix). XGBoost loads all data upfront, so Lance advantage is in step 1 only.

#### Benchmark 5: PyTorch MLP Volume Model (end-to-end)

- **Dataset**: Hourly events table (13M rows)
- **Steps timed individually**:
  1. `data_load` — read from MinIO / initialize dataset
  2. `dataloader_init` — create DataLoader with batching + shuffling
  3. `train_5_epochs` — train MLP for 5 epochs (batch_size=1024, shuffle=True)
  4. `total` — end-to-end
- **Lance approach**: use `lance.torch.data.LanceDataset` — provides native random-access iteration without loading full dataset to memory. DataLoader pulls batches directly from Lance's indexed storage.
- **Parquet approach**: `pq.read_table()` → numpy array → `torch.TensorDataset` + `DataLoader(shuffle=True)`. Must load entire 13M rows into memory first.
- **MLP architecture** (simple, fixed):
  ```python
  nn.Sequential(
      nn.Linear(5, 64),   # 5 input features (hour, atg_start, atg_diff + 2 embeddings)
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),   # predict next delivery volume
  )
  ```
- **Expected outcome**: Major difference. Lance avoids the full-table memory load and provides efficient shuffled access for each epoch. Parquet must hold 13M rows in memory throughout training. Memory difference should be dramatic (Lance ~batch-sized buffer vs Parquet ~full dataset). Training wall-clock also faster since Lance doesn't need the upfront load.

---

## 6. Metrics Collection

### Instrumentation approach

| Metric | Tool | How |
|--------|------|-----|
| Wall-clock time | `time.perf_counter()` | Wrap each operation |
| Peak memory | `tracemalloc.get_traced_memory()` | Start/stop around each benchmark |
| Throughput | Derived | rows / wall_clock, bytes / wall_clock |
| First-row latency | `time.perf_counter()` | Time until first row/batch is returned |
| File size on disk | `boto3 head_object` | After upload, query total size per format |
| S3 API call count | Proxy wrapper on boto3/fsspec | Increment counter on each S3 operation |
| CPU utilization | `psutil.Process().cpu_percent(interval=None)` | Sample at start/end of each benchmark |

### S3 call counting

Wrap the S3 filesystem with a counting proxy:

```python
class S3CallCounter:
    def __init__(self):
        self.calls = {"GetObject": 0, "HeadObject": 0, "ListObjectsV2": 0}

    def reset(self):
        self.calls = {k: 0 for k in self.calls}

    @property
    def total(self):
        return sum(self.calls.values())
```

Hook into `botocore`'s event system (`before-call` event) to count actual S3 API calls made by both PyArrow and Lance.

### Output format

All metrics stored in a structured dict:

```python
{
    "environment": { "python_version": ..., "packages": {...}, "cpu": ..., "ram": ... },
    "benchmarks": {
        "random_access": {
            "lance": { "wall_clock_mean": ..., "wall_clock_std": ..., "peak_memory_mb": ..., ... },
            "parquet": { ... }
        },
        ...
    },
    "file_sizes": { "lance_decision_mb": ..., "parquet_decision_mb": ..., ... }
}
```

Saved as `results.json` alongside `report.html` for programmatic access.

---

## 7. HTML Report

Self-contained HTML file generated with Plotly. No external dependencies to view.

### Sections

1. **Summary Card** — headline: "Lance was Nx faster at random access, used Y% less memory for PyTorch training"
2. **Access Pattern Comparison** — grouped bar chart (Lance blue, Parquet orange) for each of the 3 access patterns, showing wall-clock time
3. **Training Step Breakdown** — stacked horizontal bar chart showing time per step (data_load, convert, train) for XGBoost and PyTorch × Lance/Parquet
4. **Memory Comparison** — bar chart showing peak memory per benchmark
5. **CPU Utilization** — bar chart showing average CPU % per benchmark
6. **Storage Efficiency** — table showing file sizes and S3 call counts
7. **Throughput** — bar chart (rows/sec) for sequential and column-subset tests
8. **Raw Data Table** — sortable HTML table with all metrics, mean ± std
9. **Environment** — Python version, package versions, CPU/RAM specs, OS, MinIO version

### Chart library

Plotly — generates self-contained HTML with embedded JavaScript. No server needed, works offline, interactive (hover for exact values).

---

## 8. Dependencies

```toml
[project]
name = "lance-parquet-benchmark"
requires-python = ">=3.11"
dependencies = [
    "lance>=0.20",
    "pyarrow>=17.0",
    "xgboost>=2.1",
    "torch>=2.3",
    "numpy>=1.26",
    "pandas>=2.2",
    "boto3>=1.35",
    "plotly>=5.22",
    "psutil>=5.9",
    "s3fs>=2024.6",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Managed with `uv`. All versions pinned in `uv.lock`.

---

## 9. Fairness Controls

To ensure the comparison is honest and credible:

1. **Same data** — identical DataFrames written to both formats
2. **Same partitioning** — daily partitions for both
3. **Same storage** — both read from the same MinIO instance, same bucket
4. **No local caching** — all reads go through S3 protocol, no filesystem shortcuts
5. **Warmup** — first run discarded to prime connection pools and DNS resolution
6. **GC between runs** — `gc.collect()` after each measured run to prevent memory leak contamination
7. **Fixed seeds** — data generation and random access indices are deterministic
8. **Fixed model params** — no hyperparameter tuning; same compute work for both formats
9. **Clear labeling** — report states exactly what was measured, how many repetitions, and conditions

---

## 10. Expected Results (Hypothesis)

| Benchmark | Expected Winner | Why |
|-----------|----------------|-----|
| Random row access | **Lance (10-50x)** | Native row index vs row-group scanning |
| Sequential scan | **Tie or Parquet slight edge** | Both optimized for sequential reads |
| Column subset | **Slight Lance edge** | Better column metadata locality over S3 |
| XGBoost training | **Lance (faster load step)** | Difference only in I/O step |
| PyTorch training | **Lance (significant)** | Native random-access DataLoader vs full-memory load. Memory difference dramatic. |
| File size | **Parquet slightly smaller** | Parquet has more mature compression codecs |
| S3 call count | **Lance fewer calls** | Self-describing metadata reduces listing/head calls |

These are hypotheses to validate. The benchmark may produce different results — that's the point of running it.

---

## 11. Out of Scope

- GPU benchmarks (data loading is CPU/IO-bound; GPU only matters during forward/backward pass which is identical for both formats)
- Multi-node or distributed training
- Concurrent reader benchmarks
- Iceberg integration (this benchmarks raw file format performance)
- Data update/append operations (future benchmark)
- Production deployment guidance
