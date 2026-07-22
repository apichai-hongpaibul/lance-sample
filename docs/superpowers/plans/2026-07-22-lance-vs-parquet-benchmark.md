# Lance vs Parquet Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained benchmark comparing Lance vs Parquet file formats for AI model training data loading via MinIO, producing an HTML report with performance metrics.

**Architecture:** Single Python CLI project with docker-compose for MinIO. One entry point (`python -m src.benchmark`) generates synthetic GAIA data, writes both formats to MinIO, runs 5 benchmarks (3 access patterns + 2 model training), and produces an HTML report with Plotly charts.

**Tech Stack:** Python 3.11+, uv, lance, pyarrow, xgboost, torch, numpy, pandas, boto3, s3fs, plotly, psutil, Docker (MinIO)

## Global Constraints

- Python >=3.11
- Package management: `uv`
- All reads go through S3 protocol to MinIO (no local file shortcuts)
- Random seed: 42 for all data generation and index sampling
- MinIO image: `minio/minio:RELEASE.2024-06-13T22-53-53Z`
- MinIO credentials: `minioadmin` / `minioadmin`
- Bucket name: `benchmark`
- Benchmark repetitions: 1 warmup + 3 measured
- `gc.collect()` between benchmark runs

---

## File Map

| File | Responsibility |
|------|---------------|
| `docker-compose.yml` | MinIO container definition |
| `pyproject.toml` | Project metadata + dependencies |
| `run.sh` | One-command entry point |
| `src/__init__.py` | Package marker |
| `src/config.py` | All constants: MinIO config, data scale, column lists |
| `src/generate.py` | Synthetic data generation for both datasets |
| `src/storage.py` | Write Lance/Parquet to MinIO, query file sizes |
| `src/metrics.py` | Timing, memory, CPU, S3 call counter utilities |
| `src/benchmark.py` | Orchestrator: generate → benchmark → report (also `__main__`) |
| `src/benchmarks/__init__.py` | Package marker |
| `src/benchmarks/access_patterns.py` | Random access, sequential scan, column subset benchmarks |
| `src/benchmarks/training.py` | XGBoost and PyTorch training benchmarks |
| `src/models/__init__.py` | Package marker |
| `src/models/decision_xgb.py` | XGBoost Decision Model wrapper |
| `src/models/volume_mlp.py` | PyTorch MLP Volume Model wrapper |
| `src/report.py` | HTML report generation with Plotly |
| `tests/test_config.py` | Config unit tests |
| `tests/test_generate.py` | Data generation tests |
| `tests/test_metrics.py` | Metrics utilities tests |
| `tests/test_storage.py` | Storage integration tests (requires MinIO) |
| `tests/test_benchmarks.py` | Benchmark logic tests (small scale) |
| `tests/test_report.py` | Report generation tests |

---

### Task 1: Project Scaffolding & Infrastructure

**Files:**
- Create: `docker-compose.yml`
- Create: `pyproject.toml`
- Create: `run.sh`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `src.config.Settings` dataclass with fields: `minio_endpoint`, `minio_access_key`, `minio_secret_key`, `bucket_name`, `lance_decision_path`, `parquet_decision_path`, `lance_volume_path`, `parquet_volume_path`, `num_sites`, `num_tanks_per_site`, `num_days`, `seed`, `benchmark_repeats`, `warmup_runs`

- [ ] **Step 1: Create docker-compose.yml**

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

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "lance-parquet-benchmark"
version = "0.1.0"
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

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create run.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "Starting MinIO..."
docker compose up -d --wait

echo "Installing dependencies..."
uv sync

echo "Running benchmark..."
uv run python -m src.benchmark

echo "Done! Open report.html to view results."
```

- [ ] **Step 4: Create src/__init__.py**

```python
"""Lance vs Parquet benchmark package."""
```

- [ ] **Step 5: Write failing test for config**

Create `tests/test_config.py`:

```python
from src.config import Settings


def test_settings_defaults():
    s = Settings()
    assert s.minio_endpoint == "http://localhost:9000"
    assert s.minio_access_key == "minioadmin"
    assert s.minio_secret_key == "minioadmin"
    assert s.bucket_name == "benchmark"
    assert s.num_sites == 500
    assert s.num_tanks_per_site == 3
    assert s.num_days == 365
    assert s.seed == 42
    assert s.benchmark_repeats == 3
    assert s.warmup_runs == 1


def test_settings_paths():
    s = Settings()
    assert s.lance_decision_path == "s3://benchmark/lance/decision"
    assert s.parquet_decision_path == "s3://benchmark/parquet/decision"
    assert s.lance_volume_path == "s3://benchmark/lance/volume"
    assert s.parquet_volume_path == "s3://benchmark/parquet/volume"


def test_settings_storage_options():
    s = Settings()
    opts = s.storage_options
    assert opts["aws_access_key_id"] == "minioadmin"
    assert opts["aws_secret_access_key"] == "minioadmin"
    assert opts["endpoint_url"] == "http://localhost:9000"
    assert opts["allow_http"] == "true"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 7: Implement src/config.py**

```python
"""Project configuration and constants."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """All benchmark settings in one place."""

    # MinIO connection
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    bucket_name: str = "benchmark"

    # S3 paths
    lance_decision_path: str = "s3://benchmark/lance/decision"
    parquet_decision_path: str = "s3://benchmark/parquet/decision"
    lance_volume_path: str = "s3://benchmark/lance/volume"
    parquet_volume_path: str = "s3://benchmark/parquet/volume"

    # Data scale
    num_sites: int = 500
    num_tanks_per_site: int = 3
    num_days: int = 365
    seed: int = 42

    # Benchmark params
    benchmark_repeats: int = 3
    warmup_runs: int = 1

    # Column subsets for benchmark 3
    column_subset: tuple[str, ...] = (
        "overdue_ratio",
        "inv_days_cover",
        "dow_hist_rate",
        "hist_rate",
        "avg_sale_7d",
        "open_inventory",
        "tank_capacity",
        "delivery_occurred",
    )

    # Random access params
    random_access_sample_size: int = 10_000

    @property
    def storage_options(self) -> dict[str, str]:
        """S3 storage options for lance and pyarrow."""
        return {
            "aws_access_key_id": self.minio_access_key,
            "aws_secret_access_key": self.minio_secret_key,
            "endpoint_url": self.minio_endpoint,
            "allow_http": "true",
            "region": "us-east-1",
        }

    @property
    def total_decision_rows(self) -> int:
        return self.num_sites * self.num_tanks_per_site * self.num_days

    @property
    def total_volume_rows(self) -> int:
        return self.num_sites * self.num_tanks_per_site * self.num_days * 24
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: All 3 tests PASS

- [ ] **Step 9: Make run.sh executable and commit**

```bash
chmod +x run.sh
git add docker-compose.yml pyproject.toml run.sh src/__init__.py src/config.py tests/test_config.py
git commit -m "feat: project scaffolding — docker-compose, pyproject, config"
```

---

### Task 2: Project Documentation (AGENTS.md & README.md)

**Files:**
- Create: `AGENTS.md`
- Create: `README.md`

**Interfaces:**
- Consumes: `src.config.Settings` (for documenting project constants)
- Produces: Project documentation for contributors and users

- [ ] **Step 1: Create AGENTS.md**

```markdown
# AGENTS.md

## Project Overview

Lance vs Parquet Benchmark — an internal PoC comparing Lance and Parquet file formats for AI model training data loading via S3-compatible storage (MinIO).

## Architecture

Single Python CLI project. One entry point (`python -m src.benchmark`) that:
1. Generates synthetic GAIA fuel-delivery feature data
2. Writes identical data in both Lance and Parquet formats to MinIO
3. Runs 5 benchmarks (3 access patterns + 2 model training)
4. Produces a self-contained HTML report with Plotly charts

## Tech Stack

- **Language:** Python 3.11+
- **Package manager:** uv
- **Storage:** MinIO (Docker, S3-compatible)
- **Data formats:** Lance (`lance`), Parquet (`pyarrow`)
- **ML:** XGBoost, PyTorch
- **Metrics:** tracemalloc, psutil, time.perf_counter
- **Reporting:** Plotly (self-contained HTML)
- **Testing:** pytest

## Key Files

| File | Purpose |
|------|---------|
| `src/config.py` | All constants: MinIO config, data scale, column lists |
| `src/generate.py` | Synthetic data generation (Decision + Volume datasets) |
| `src/storage.py` | Write Lance/Parquet to MinIO, query file sizes |
| `src/metrics.py` | Timing, memory, CPU, S3 call counter utilities |
| `src/benchmark.py` | Orchestrator + CLI entry point (`__main__`) |
| `src/benchmarks/access_patterns.py` | Random access, sequential scan, column subset |
| `src/benchmarks/training.py` | XGBoost and PyTorch training benchmarks |
| `src/models/decision_xgb.py` | XGBoost Decision Model wrapper |
| `src/models/volume_mlp.py` | PyTorch MLP Volume Model wrapper |
| `src/report.py` | HTML report generation |

## Commands

```bash
# Start infrastructure
docker compose up -d --wait

# Install dependencies
uv sync

# Run full benchmark
uv run python -m src.benchmark

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src

# One-command run (start MinIO + benchmark)
./run.sh
```

## Data Scale

- Decision Model: ~547,500 rows (500 sites × 3 tanks × 365 days), 30+ columns
- Volume Model: ~13,140,000 rows (500 sites × 3 tanks × 365 days × 24 hours), 6 columns

## Benchmarks

1. **Random Row Access** — 10K random rows from Decision dataset
2. **Sequential Scan** — full table read of Decision dataset
3. **Column Subset** — 8 of 30+ columns from Decision dataset
4. **XGBoost Training** — end-to-end with step timing (Decision Model)
5. **PyTorch Training** — end-to-end with step timing (Volume Model, 13M rows)

## Conventions

- All S3 reads go through MinIO (no local file shortcuts)
- Seed 42 for all random operations
- 1 warmup + 3 measured repetitions per benchmark
- `gc.collect()` between runs
- No GPU benchmarks (data loading is CPU/IO-bound)
```

- [ ] **Step 2: Create README.md**

```markdown
# Lance vs Parquet Benchmark

A self-contained benchmark comparing **Lance** and **Parquet** file formats for AI model training data loading via MinIO (S3-compatible storage).

Built with realistic synthetic data from the GAIA fuel-delivery platform to demonstrate where Lance outperforms Parquet — particularly in random-access patterns used by ML training pipelines.

## Quick Start

```bash
# Prerequisites: Docker, uv (https://docs.astral.sh/uv/)
./run.sh
```

Or step by step:

```bash
docker compose up -d --wait     # Start MinIO
uv sync                         # Install Python deps
uv run python -m src.benchmark  # Run benchmarks
open report.html                # View results
```

## What It Measures

### Access Patterns
| Benchmark | What | Why It Matters |
|-----------|------|---------------|
| Random Row Access | Read 10K random rows | PyTorch DataLoader with shuffle |
| Sequential Scan | Read full table (547K rows) | Batch feature extraction |
| Column Subset | Read 8/30+ columns | Feature selection for training |

### Model Training (end-to-end with step timing)
| Benchmark | Model | Dataset |
|-----------|-------|---------|
| XGBoost | Decision Model (binary classification) | 547K rows, 30+ features |
| PyTorch MLP | Volume Model (regression) | 13M rows, 6 features |

### Metrics Collected
- Wall-clock time (mean ± std over 3 runs)
- Peak memory usage (MB)
- CPU utilization (%)
- Throughput (rows/sec, MB/sec)
- First-row latency
- S3 API call count
- File size on disk

## Output

- `report.html` — self-contained HTML report with interactive Plotly charts
- `results.json` — raw metrics for programmatic access

## Why Lance?

| Scenario | Lance | Parquet |
|----------|-------|---------|
| Random row access | O(1) via native index | Must scan row groups |
| Shuffled DataLoader | Stream from disk, batch-sized memory | Load entire dataset to RAM first |
| Column projection over S3 | Efficient metadata locality | Good but more S3 calls |
| Sequential scan | Comparable | Comparable (mature implementation) |

## Data

Synthetic data mirrors the GAIA fuel-delivery platform's Gold feature tables:

- **Decision Model features** — 500 sites × 3 tanks × 365 days = 547K rows with 30+ columns (overdue_ratio, inv_days_cover, hist_rate, etc.)
- **Hourly ATG events** — same sites/tanks × 24 hours = 13.1M rows with 6 columns

Data has realistic correlations (not random noise) so model training is meaningful.

## Requirements

- Docker & Docker Compose
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- ~2GB free disk (MinIO data + generated datasets)
- ~4GB RAM (for 13M row Parquet-in-memory baseline)

## Project Structure

```
├── docker-compose.yml          # MinIO container
├── pyproject.toml              # Dependencies (uv-managed)
├── run.sh                      # One-command entry point
├── src/
│   ├── config.py               # All settings and constants
│   ├── generate.py             # Synthetic data generator
│   ├── storage.py              # MinIO read/write for both formats
│   ├── metrics.py              # Instrumentation utilities
│   ├── benchmark.py            # Main orchestrator
│   ├── benchmarks/
│   │   ├── access_patterns.py  # 3 access pattern benchmarks
│   │   └── training.py         # 2 training benchmarks
│   ├── models/
│   │   ├── decision_xgb.py    # XGBoost classifier
│   │   └── volume_mlp.py      # PyTorch MLP regressor
│   └── report.py              # HTML report with Plotly
├── tests/                      # pytest test suite
├── ref/                        # GAIA reference docs
└── docs/                       # Design specs and plans
```

## License

Internal use only.
```

- [ ] **Step 3: Commit documentation**

```bash
git add AGENTS.md README.md
git commit -m "docs: add AGENTS.md and README.md with project overview"
```

---

### Task 3: Metrics Utilities

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `BenchmarkTimer` context manager: `__enter__` → starts timer, `__exit__` → records elapsed. Property `elapsed_sec: float`
  - `MemoryTracker` context manager: `__enter__` → starts tracemalloc, `__exit__` → records peak. Property `peak_mb: float`
  - `CpuTracker` context manager: uses psutil, property `cpu_percent: float`
  - `S3CallCounter` class: methods `reset()`, `start()`, `stop()`, property `total: int`, property `calls: dict[str, int]`
  - `BenchmarkResult` dataclass: fields `wall_clock_sec`, `peak_memory_mb`, `cpu_percent`, `s3_calls`, `rows_read`, `bytes_read`, `first_row_latency_sec` (all Optional[float/int])
  - `run_benchmark(fn, repeats, warmup) -> list[BenchmarkResult]`: runs function with warmup + repeats, returns list of results

- [ ] **Step 1: Write failing tests**

Create `tests/test_metrics.py`:

```python
import time

from src.metrics import (
    BenchmarkResult,
    BenchmarkTimer,
    CpuTracker,
    MemoryTracker,
    S3CallCounter,
    run_benchmark,
)


def test_benchmark_timer():
    with BenchmarkTimer() as t:
        time.sleep(0.05)
    assert 0.04 < t.elapsed_sec < 0.2


def test_memory_tracker():
    with MemoryTracker() as m:
        data = [0] * 100_000
    assert m.peak_mb > 0


def test_cpu_tracker():
    with CpuTracker() as c:
        sum(range(1_000_000))
    # cpu_percent can be 0 on very fast operations, just check it's a number
    assert isinstance(c.cpu_percent, float)


def test_s3_call_counter_manual():
    counter = S3CallCounter()
    counter.reset()
    # Simulate incrementing (actual botocore hooking tested in integration)
    counter.calls["GetObject"] += 5
    counter.calls["HeadObject"] += 2
    assert counter.total == 7


def test_benchmark_result_dataclass():
    r = BenchmarkResult(wall_clock_sec=1.5, peak_memory_mb=100.0)
    assert r.wall_clock_sec == 1.5
    assert r.peak_memory_mb == 100.0
    assert r.cpu_percent is None
    assert r.s3_calls is None


def test_run_benchmark():
    call_count = 0

    def dummy_fn():
        nonlocal call_count
        call_count += 1
        time.sleep(0.01)
        return BenchmarkResult(wall_clock_sec=0.01, peak_memory_mb=1.0)

    results = run_benchmark(dummy_fn, repeats=3, warmup=1)
    assert len(results) == 3
    assert call_count == 4  # 1 warmup + 3 repeats
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.metrics'`

- [ ] **Step 3: Implement src/metrics.py**

```python
"""Benchmark instrumentation: timing, memory, CPU, S3 call tracking."""

import gc
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, Optional

import psutil


class BenchmarkTimer:
    """Context manager for wall-clock timing."""

    def __init__(self):
        self.elapsed_sec: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_sec = time.perf_counter() - self._start
        return False


class MemoryTracker:
    """Context manager for peak memory tracking via tracemalloc."""

    def __init__(self):
        self.peak_mb: float = 0.0

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, *exc):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mb = peak / (1024 * 1024)
        return False


class CpuTracker:
    """Context manager for CPU utilization tracking."""

    def __init__(self):
        self.cpu_percent: float = 0.0
        self._process = psutil.Process()

    def __enter__(self):
        self._process.cpu_percent()  # first call initializes measurement
        return self

    def __exit__(self, *exc):
        self.cpu_percent = self._process.cpu_percent()
        return False


class S3CallCounter:
    """Counts S3 API calls by hooking into botocore events."""

    def __init__(self):
        self.calls: dict[str, int] = {
            "GetObject": 0,
            "HeadObject": 0,
            "ListObjectsV2": 0,
            "ListObjects": 0,
        }
        self._session = None

    def reset(self):
        self.calls = {k: 0 for k in self.calls}

    @property
    def total(self) -> int:
        return sum(self.calls.values())

    def _event_handler(self, event_name, **kwargs):
        """Botocore before-call event handler."""
        params = kwargs.get("params", {})
        # Extract operation from the url/headers or event name
        for op in self.calls:
            if op in event_name:
                self.calls[op] += 1
                return

    def start(self):
        """Start counting. Call before benchmark operations."""
        self.reset()

    def stop(self):
        """Stop counting. Call after benchmark operations."""
        pass  # counts are already accumulated


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    wall_clock_sec: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_percent: Optional[float] = None
    s3_calls: Optional[int] = None
    s3_calls_detail: Optional[dict[str, int]] = None
    rows_read: Optional[int] = None
    bytes_read: Optional[int] = None
    first_row_latency_sec: Optional[float] = None
    throughput_rows_per_sec: Optional[float] = None
    throughput_mb_per_sec: Optional[float] = None


def run_benchmark(
    fn: Callable[[], BenchmarkResult],
    repeats: int = 3,
    warmup: int = 1,
) -> list[BenchmarkResult]:
    """Run a benchmark function with warmup and repetitions.

    Args:
        fn: Callable that runs the benchmark and returns a BenchmarkResult.
        repeats: Number of measured repetitions.
        warmup: Number of warmup runs (discarded).

    Returns:
        List of BenchmarkResult from the measured runs only.
    """
    # Warmup runs
    for _ in range(warmup):
        fn()
        gc.collect()

    # Measured runs
    results = []
    for _ in range(repeats):
        result = fn()
        results.append(result)
        gc.collect()

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: metrics utilities — timer, memory, CPU, S3 counter"
```

---

### Task 4: Data Generation

**Files:**
- Create: `src/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `src.config.Settings` (num_sites, num_tanks_per_site, num_days, seed)
- Produces:
  - `generate_decision_features(settings: Settings) -> pa.Table`: returns PyArrow Table with 547K rows, 32 columns
  - `generate_volume_events(settings: Settings) -> pa.Table`: returns PyArrow Table with 13.1M rows, 6 columns

- [ ] **Step 1: Write failing tests**

Create `tests/test_generate.py`:

```python
import pyarrow as pa

from src.config import Settings
from src.generate import generate_decision_features, generate_volume_events


def test_decision_features_shape():
    # Use small scale for testing
    s = Settings(num_sites=5, num_tanks_per_site=2, num_days=3)
    table = generate_decision_features(s)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 5 * 2 * 3  # 30 rows
    assert table.num_columns >= 30


def test_decision_features_columns():
    s = Settings(num_sites=2, num_tanks_per_site=1, num_days=2)
    table = generate_decision_features(s)
    col_names = table.column_names
    assert "site_code" in col_names
    assert "tank_id" in col_names
    assert "feature_date" in col_names
    assert "overdue_ratio" in col_names
    assert "delivery_occurred" in col_names
    assert "inv_days_cover" in col_names
    assert "dow_hist_rate" in col_names


def test_decision_features_reproducible():
    s = Settings(num_sites=3, num_tanks_per_site=1, num_days=2)
    t1 = generate_decision_features(s)
    t2 = generate_decision_features(s)
    assert t1.equals(t2)


def test_decision_features_target_distribution():
    s = Settings(num_sites=10, num_tanks_per_site=2, num_days=30)
    table = generate_decision_features(s)
    target = table.column("delivery_occurred").to_pylist()
    # Should have both 0s and 1s
    assert 0 in target
    assert 1 in target


def test_volume_events_shape():
    s = Settings(num_sites=3, num_tanks_per_site=2, num_days=2)
    table = generate_volume_events(s)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 3 * 2 * 2 * 24  # 288 rows
    assert table.num_columns == 6


def test_volume_events_columns():
    s = Settings(num_sites=2, num_tanks_per_site=1, num_days=1)
    table = generate_volume_events(s)
    expected_cols = {"reading_date", "reading_hour", "site_code", "tank_id", "atg_start", "atg_diff"}
    assert set(table.column_names) == expected_cols


def test_volume_events_hour_range():
    s = Settings(num_sites=1, num_tanks_per_site=1, num_days=1)
    table = generate_volume_events(s)
    hours = table.column("reading_hour").to_pylist()
    assert min(hours) == 0
    assert max(hours) == 23


def test_volume_events_reproducible():
    s = Settings(num_sites=2, num_tanks_per_site=1, num_days=2)
    t1 = generate_volume_events(s)
    t2 = generate_volume_events(s)
    assert t1.equals(t2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.generate'`

- [ ] **Step 3: Implement src/generate.py**

```python
"""Synthetic data generation for GAIA-style feature tables."""

import datetime

import numpy as np
import pyarrow as pa

from src.config import Settings

# Products and regions from the GAIA spec
PRODUCTS = ["5000018", "5000011", "5000012"]  # HSD, ULG91, ULG95
REGIONS = ["NORTH", "SOUTH", "EAST", "WEST", "CENTRAL"]


def generate_decision_features(settings: Settings) -> pa.Table:
    """Generate the Decision Model feature table (wide, 30+ columns).

    Produces realistic correlated features matching the GAIA Gold
    gold_decision_model_features schema.
    """
    rng = np.random.default_rng(settings.seed)

    n_rows = settings.total_decision_rows
    n_sites = settings.num_sites
    n_tanks = settings.num_tanks_per_site
    n_days = settings.num_days

    # Generate keys by tiling
    site_codes = [f"I{i:03d}" for i in range(1, n_sites + 1)]
    tank_ids = [f"T{t:02d}" for t in range(1, n_tanks + 1)]

    # Create all combinations: site × tank × day
    start_date = datetime.date(2025, 7, 22)
    dates = [start_date + datetime.timedelta(days=d) for d in range(n_days)]

    # Build arrays by repeating patterns
    site_arr = np.repeat(site_codes, n_tanks * n_days)
    tank_arr = np.tile(np.repeat(tank_ids, n_days), n_sites)
    date_arr = np.tile(dates, n_sites * n_tanks)

    # Assign fixed attributes per site
    site_regions = rng.choice(REGIONS, size=n_sites)
    region_arr = np.repeat(site_regions, n_tanks * n_days)

    site_products = rng.choice(PRODUCTS, size=n_sites)
    product_arr = np.repeat(site_products, n_tanks * n_days)

    # Tank capacity (fixed per site-tank, but we generate per row for simplicity)
    tank_capacities = rng.uniform(15000, 40000, size=n_sites * n_tanks)
    tank_capacity_arr = np.repeat(tank_capacities, n_days)

    # Generate correlated features
    # Base delivery probability influenced by overdue_ratio
    base_overdue = rng.exponential(0.8, size=n_rows)
    overdue_ratio = np.clip(base_overdue, 0.0, 3.0)

    # Higher overdue → more likely to deliver
    delivery_prob = 1 / (1 + np.exp(-(overdue_ratio - 1.0)))
    delivery_occurred = rng.binomial(1, delivery_prob).astype(np.int8)

    # Inventory features (anti-correlated with overdue)
    inv_days_cover = np.clip(rng.normal(7.0, 3.0, size=n_rows) - overdue_ratio * 2, 0.0, 15.0)
    open_inventory = tank_capacity_arr * rng.uniform(0.1, 0.9, size=n_rows)

    # Sales features
    avg_sale_7d = rng.uniform(500, 5000, size=n_rows)
    avg_sale_30d = avg_sale_7d * rng.uniform(0.8, 1.2, size=n_rows)

    # Historical rates with day-of-week seasonality
    day_of_week = np.array([d.weekday() for d in date_arr], dtype=np.int8)
    # Weekend has lower delivery rate
    weekend_factor = np.where((day_of_week == 5) | (day_of_week == 6), 0.3, 0.7)
    dow_hist_rate = np.clip(weekend_factor + rng.normal(0, 0.1, size=n_rows), 0.0, 1.0)
    hist_rate = np.clip(rng.beta(2, 3, size=n_rows), 0.0, 1.0)

    # Delivery volume features
    delivery_sum_28d = rng.uniform(0, 200000, size=n_rows)
    proj_end_fill_ratio = np.clip(rng.beta(3, 2, size=n_rows), 0.0, 1.0)

    # Percentile features
    group_overdue_pct = rng.uniform(0.0, 1.0, size=n_rows)
    group_cover_pct = rng.uniform(0.0, 1.0, size=n_rows)

    # Other features
    last_delivery_days_ago = rng.integers(0, 31, size=n_rows).astype(np.int32)
    intransit_volume = rng.uniform(0, 30000, size=n_rows)
    usage_day_at_approval = rng.uniform(0.5, 10.0, size=n_rows)
    current_inventory_at_approval = open_inventory * rng.uniform(0.8, 1.0, size=n_rows)

    # Date features
    day_of_month = np.array([d.day for d in date_arr], dtype=np.int8)
    month = np.array([d.month for d in date_arr], dtype=np.int8)
    is_weekend = ((day_of_week == 5) | (day_of_week == 6))
    is_holiday = rng.random(size=n_rows) < 0.03  # ~3% holidays

    # Lag features
    delivery_volume_lag1 = rng.uniform(0, 30000, size=n_rows)
    delivery_volume_lag7 = rng.uniform(0, 30000, size=n_rows)
    rolling_delivery_count_7d = rng.integers(0, 8, size=n_rows).astype(np.int32)
    rolling_delivery_count_30d = rng.integers(0, 31, size=n_rows).astype(np.int32)

    # Pair features
    pair_deliv_mean = rng.uniform(5000, 25000, size=n_rows)
    yoy_deliv = rng.uniform(0, 50000, size=n_rows)

    # Build PyArrow Table
    table = pa.table({
        "site_code": pa.array(site_arr, type=pa.string()),
        "tank_id": pa.array(tank_arr, type=pa.string()),
        "feature_date": pa.array(date_arr, type=pa.date32()),
        "product_code": pa.array(product_arr, type=pa.string()),
        "region_code": pa.array(region_arr, type=pa.string()),
        "tank_capacity": pa.array(tank_capacity_arr, type=pa.float64()),
        "open_inventory": pa.array(open_inventory, type=pa.float64()),
        "avg_sale_7d": pa.array(avg_sale_7d, type=pa.float64()),
        "avg_sale_30d": pa.array(avg_sale_30d, type=pa.float64()),
        "overdue_ratio": pa.array(overdue_ratio, type=pa.float64()),
        "inv_days_cover": pa.array(inv_days_cover, type=pa.float64()),
        "dow_hist_rate": pa.array(dow_hist_rate, type=pa.float64()),
        "hist_rate": pa.array(hist_rate, type=pa.float64()),
        "delivery_sum_28d": pa.array(delivery_sum_28d, type=pa.float64()),
        "proj_end_fill_ratio": pa.array(proj_end_fill_ratio, type=pa.float64()),
        "group_overdue_pct": pa.array(group_overdue_pct, type=pa.float64()),
        "group_cover_pct": pa.array(group_cover_pct, type=pa.float64()),
        "last_delivery_days_ago": pa.array(last_delivery_days_ago, type=pa.int32()),
        "intransit_volume": pa.array(intransit_volume, type=pa.float64()),
        "usage_day_at_approval": pa.array(usage_day_at_approval, type=pa.float64()),
        "current_inventory_at_approval": pa.array(current_inventory_at_approval, type=pa.float64()),
        "day_of_week": pa.array(day_of_week, type=pa.int8()),
        "day_of_month": pa.array(day_of_month, type=pa.int8()),
        "month": pa.array(month, type=pa.int8()),
        "is_weekend": pa.array(is_weekend, type=pa.bool_()),
        "is_holiday": pa.array(is_holiday, type=pa.bool_()),
        "delivery_volume_lag1": pa.array(delivery_volume_lag1, type=pa.float64()),
        "delivery_volume_lag7": pa.array(delivery_volume_lag7, type=pa.float64()),
        "rolling_delivery_count_7d": pa.array(rolling_delivery_count_7d, type=pa.int32()),
        "rolling_delivery_count_30d": pa.array(rolling_delivery_count_30d, type=pa.int32()),
        "pair_deliv_mean": pa.array(pair_deliv_mean, type=pa.float64()),
        "yoy_deliv": pa.array(yoy_deliv, type=pa.float64()),
        "delivery_occurred": pa.array(delivery_occurred, type=pa.int8()),
    })
    return table


def generate_volume_events(settings: Settings) -> pa.Table:
    """Generate the Volume Model hourly events table (narrow, time-series).

    Produces ATG hourly readings with realistic daily consumption patterns.
    """
    rng = np.random.default_rng(settings.seed + 1)  # different seed from decision

    n_sites = settings.num_sites
    n_tanks = settings.num_tanks_per_site
    n_days = settings.num_days
    n_rows = settings.total_volume_rows

    # Keys
    site_codes = [f"I{i:03d}" for i in range(1, n_sites + 1)]
    tank_ids = [f"T{t:02d}" for t in range(1, n_tanks + 1)]
    start_date = datetime.date(2025, 7, 22)
    dates = [start_date + datetime.timedelta(days=d) for d in range(n_days)]
    hours = list(range(24))

    # Build arrays: site × tank × day × hour
    site_arr = np.repeat(site_codes, n_tanks * n_days * 24)
    tank_arr = np.tile(np.repeat(tank_ids, n_days * 24), n_sites)
    date_arr = np.tile(np.repeat(dates, 24), n_sites * n_tanks)
    hour_arr = np.tile(hours, n_sites * n_tanks * n_days).astype(np.int8)

    # ATG consumption pattern: peak at 7-9 and 17-19
    hour_weights = np.array([
        0.2, 0.1, 0.1, 0.1, 0.2, 0.3,  # 0-5: night, low
        0.5, 0.9, 1.0, 0.8, 0.7, 0.6,  # 6-11: morning peak
        0.5, 0.5, 0.5, 0.6, 0.7, 0.9,  # 12-17: afternoon rise
        1.0, 0.8, 0.6, 0.4, 0.3, 0.2,  # 18-23: evening peak then decline
    ])
    base_consumption = hour_weights[hour_arr] * rng.uniform(50, 300, size=n_rows)
    atg_diff = np.abs(base_consumption + rng.normal(0, 20, size=n_rows))

    # ATG start: tank level that decreases through the day
    # Simplified: random starting level per site-tank-day, minus cumulative consumption
    tank_capacity_per_combo = rng.uniform(15000, 40000, size=n_sites * n_tanks)
    daily_start = np.repeat(tank_capacity_per_combo, n_days * 24) * rng.uniform(0.3, 0.9, size=n_rows)
    atg_start = np.clip(daily_start - atg_diff * (hour_arr / 24.0), 1000, 40000)

    table = pa.table({
        "reading_date": pa.array(date_arr, type=pa.date32()),
        "reading_hour": pa.array(hour_arr, type=pa.int8()),
        "site_code": pa.array(site_arr, type=pa.string()),
        "tank_id": pa.array(tank_arr, type=pa.string()),
        "atg_start": pa.array(atg_start, type=pa.float64()),
        "atg_diff": pa.array(atg_diff, type=pa.float64()),
    })
    return table
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/generate.py tests/test_generate.py
git commit -m "feat: synthetic data generation — decision features + volume events"
```

---

### Task 5: Storage Layer (Write/Read both formats to MinIO)

**Files:**
- Create: `src/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: `src.config.Settings` (storage_options, paths), `pyarrow.Table`
- Produces:
  - `write_parquet_to_minio(table: pa.Table, path: str, settings: Settings, partition_col: str | None = "feature_date") -> None`
  - `write_lance_to_minio(table: pa.Table, path: str, settings: Settings) -> None`
  - `ensure_bucket(settings: Settings) -> None`: creates the MinIO bucket if it doesn't exist
  - `get_dataset_size_mb(path: str, settings: Settings) -> float`: returns total size of dataset in MB
  - `get_lance_dataset(path: str, settings: Settings) -> lance.LanceDataset`
  - `get_parquet_dataset(path: str, settings: Settings) -> pq.ParquetDataset`

- [ ] **Step 1: Write failing tests**

Create `tests/test_storage.py`:

```python
"""Storage tests — require MinIO running (docker compose up -d)."""

import pyarrow as pa
import pytest

from src.config import Settings
from src.storage import (
    ensure_bucket,
    get_dataset_size_mb,
    get_lance_dataset,
    get_parquet_dataset,
    write_lance_to_minio,
    write_parquet_to_minio,
)

# Use test-specific paths to avoid collision
TEST_SETTINGS = Settings(
    lance_decision_path="s3://benchmark/test/lance/decision",
    parquet_decision_path="s3://benchmark/test/parquet/decision",
)


@pytest.fixture(scope="module")
def sample_table():
    """Small PyArrow table for testing."""
    import datetime

    return pa.table({
        "site_code": ["I001", "I001", "I002", "I002"],
        "tank_id": ["T01", "T01", "T01", "T01"],
        "feature_date": [
            datetime.date(2025, 7, 22),
            datetime.date(2025, 7, 23),
            datetime.date(2025, 7, 22),
            datetime.date(2025, 7, 23),
        ],
        "overdue_ratio": [1.5, 0.8, 2.1, 0.3],
        "delivery_occurred": pa.array([1, 0, 1, 0], type=pa.int8()),
    })


@pytest.fixture(scope="module", autouse=True)
def setup_bucket():
    """Ensure bucket exists before tests."""
    ensure_bucket(TEST_SETTINGS)


@pytest.mark.integration
def test_write_and_read_parquet(sample_table):
    path = TEST_SETTINGS.parquet_decision_path
    write_parquet_to_minio(sample_table, path, TEST_SETTINGS, partition_col="feature_date")

    ds = get_parquet_dataset(path, TEST_SETTINGS)
    result = ds.read()
    assert result.num_rows == 4


@pytest.mark.integration
def test_write_and_read_lance(sample_table):
    path = TEST_SETTINGS.lance_decision_path
    write_lance_to_minio(sample_table, path, TEST_SETTINGS)

    ds = get_lance_dataset(path, TEST_SETTINGS)
    result = ds.to_table()
    assert result.num_rows == 4


@pytest.mark.integration
def test_get_dataset_size(sample_table):
    path = TEST_SETTINGS.parquet_decision_path
    write_parquet_to_minio(sample_table, path, TEST_SETTINGS, partition_col="feature_date")
    size = get_dataset_size_mb(path, TEST_SETTINGS)
    assert size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.storage'`

- [ ] **Step 3: Implement src/storage.py**

```python
"""Storage utilities: write/read Lance and Parquet to/from MinIO."""

import lance
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import s3fs

from src.config import Settings


def _get_s3fs(settings: Settings) -> s3fs.S3FileSystem:
    """Create an S3FileSystem configured for MinIO."""
    return s3fs.S3FileSystem(
        key=settings.minio_access_key,
        secret=settings.minio_secret_key,
        endpoint_url=settings.minio_endpoint,
        use_ssl=False,
    )


def ensure_bucket(settings: Settings) -> None:
    """Create the benchmark bucket if it doesn't exist."""
    fs = _get_s3fs(settings)
    if not fs.exists(settings.bucket_name):
        fs.mkdir(settings.bucket_name)


def write_parquet_to_minio(
    table: pa.Table,
    path: str,
    settings: Settings,
    partition_col: str | None = "feature_date",
) -> None:
    """Write a PyArrow Table as a partitioned Parquet dataset to MinIO."""
    fs = _get_s3fs(settings)
    # Strip s3:// prefix for s3fs
    s3_path = path.replace("s3://", "")

    if partition_col and partition_col in table.column_names:
        pq.write_to_dataset(
            table,
            root_path=s3_path,
            partition_cols=[partition_col],
            filesystem=fs,
        )
    else:
        pq.write_table(table, f"{s3_path}/data.parquet", filesystem=fs)


def write_lance_to_minio(table: pa.Table, path: str, settings: Settings) -> None:
    """Write a PyArrow Table as a Lance dataset to MinIO."""
    storage_options = {
        "aws_access_key_id": settings.minio_access_key,
        "aws_secret_access_key": settings.minio_secret_key,
        "aws_endpoint": settings.minio_endpoint,
        "allow_http": "true",
        "aws_region": "us-east-1",
    }
    lance.write_dataset(table, path, storage_options=storage_options, mode="overwrite")


def get_lance_dataset(path: str, settings: Settings) -> lance.LanceDataset:
    """Open a Lance dataset from MinIO."""
    storage_options = {
        "aws_access_key_id": settings.minio_access_key,
        "aws_secret_access_key": settings.minio_secret_key,
        "aws_endpoint": settings.minio_endpoint,
        "allow_http": "true",
        "aws_region": "us-east-1",
    }
    return lance.dataset(path, storage_options=storage_options)


def get_parquet_dataset(path: str, settings: Settings) -> pds.Dataset:
    """Open a Parquet dataset from MinIO."""
    fs = _get_s3fs(settings)
    s3_path = path.replace("s3://", "")
    return pds.dataset(s3_path, filesystem=fs, format="parquet")


def get_dataset_size_mb(path: str, settings: Settings) -> float:
    """Get total size of all files under a path in MB."""
    fs = _get_s3fs(settings)
    s3_path = path.replace("s3://", "")
    try:
        total_bytes = sum(
            f["size"] for f in fs.ls(s3_path, detail=True) if f["type"] == "file"
        )
    except FileNotFoundError:
        # Try recursive listing for partitioned datasets
        total_bytes = 0
        for root, dirs, files in fs.walk(s3_path):
            for f in files:
                info = fs.info(f"{root}/{f}")
                total_bytes += info.get("size", 0)
    return total_bytes / (1024 * 1024)
```

- [ ] **Step 4: Run tests to verify they pass (requires MinIO running)**

Run: `docker compose up -d --wait && uv run pytest tests/test_storage.py -v -m integration`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/storage.py tests/test_storage.py
git commit -m "feat: storage layer — write/read Lance and Parquet to MinIO"
```

---

### Task 6: Access Pattern Benchmarks

**Files:**
- Create: `src/benchmarks/__init__.py`
- Create: `src/benchmarks/access_patterns.py`
- Test: `tests/test_benchmarks.py`

**Interfaces:**
- Consumes: `src.config.Settings`, `src.storage.get_lance_dataset`, `src.storage.get_parquet_dataset`, `src.metrics.BenchmarkResult`, `src.metrics.BenchmarkTimer`, `src.metrics.MemoryTracker`, `src.metrics.CpuTracker`
- Produces:
  - `bench_random_access_lance(settings: Settings) -> BenchmarkResult`
  - `bench_random_access_parquet(settings: Settings) -> BenchmarkResult`
  - `bench_sequential_scan_lance(settings: Settings) -> BenchmarkResult`
  - `bench_sequential_scan_parquet(settings: Settings) -> BenchmarkResult`
  - `bench_column_subset_lance(settings: Settings) -> BenchmarkResult`
  - `bench_column_subset_parquet(settings: Settings) -> BenchmarkResult`

- [ ] **Step 1: Write failing tests**

Create `tests/test_benchmarks.py`:

```python
"""Benchmark logic tests — use small-scale data, require MinIO."""

import pyarrow as pa
import pytest
import datetime

from src.config import Settings
from src.metrics import BenchmarkResult
from src.storage import ensure_bucket, write_lance_to_minio, write_parquet_to_minio

# Small-scale settings for testing
BENCH_SETTINGS = Settings(
    num_sites=5,
    num_tanks_per_site=2,
    num_days=3,
    random_access_sample_size=5,
    lance_decision_path="s3://benchmark/test-bench/lance/decision",
    parquet_decision_path="s3://benchmark/test-bench/parquet/decision",
)


@pytest.fixture(scope="module", autouse=True)
def setup_test_data():
    """Write small test datasets to MinIO."""
    from src.generate import generate_decision_features

    ensure_bucket(BENCH_SETTINGS)
    table = generate_decision_features(BENCH_SETTINGS)
    write_lance_to_minio(table, BENCH_SETTINGS.lance_decision_path, BENCH_SETTINGS)
    write_parquet_to_minio(
        table, BENCH_SETTINGS.parquet_decision_path, BENCH_SETTINGS, partition_col="feature_date"
    )


@pytest.mark.integration
def test_random_access_lance():
    from src.benchmarks.access_patterns import bench_random_access_lance

    result = bench_random_access_lance(BENCH_SETTINGS)
    assert isinstance(result, BenchmarkResult)
    assert result.wall_clock_sec > 0
    assert result.peak_memory_mb >= 0
    assert result.rows_read == 5


@pytest.mark.integration
def test_random_access_parquet():
    from src.benchmarks.access_patterns import bench_random_access_parquet

    result = bench_random_access_parquet(BENCH_SETTINGS)
    assert isinstance(result, BenchmarkResult)
    assert result.wall_clock_sec > 0
    assert result.rows_read == 5


@pytest.mark.integration
def test_sequential_scan_lance():
    from src.benchmarks.access_patterns import bench_sequential_scan_lance

    result = bench_sequential_scan_lance(BENCH_SETTINGS)
    assert isinstance(result, BenchmarkResult)
    assert result.wall_clock_sec > 0
    assert result.rows_read == BENCH_SETTINGS.total_decision_rows


@pytest.mark.integration
def test_sequential_scan_parquet():
    from src.benchmarks.access_patterns import bench_sequential_scan_parquet

    result = bench_sequential_scan_parquet(BENCH_SETTINGS)
    assert isinstance(result, BenchmarkResult)
    assert result.rows_read == BENCH_SETTINGS.total_decision_rows


@pytest.mark.integration
def test_column_subset_lance():
    from src.benchmarks.access_patterns import bench_column_subset_lance

    result = bench_column_subset_lance(BENCH_SETTINGS)
    assert isinstance(result, BenchmarkResult)
    assert result.wall_clock_sec > 0


@pytest.mark.integration
def test_column_subset_parquet():
    from src.benchmarks.access_patterns import bench_column_subset_parquet

    result = bench_column_subset_parquet(BENCH_SETTINGS)
    assert isinstance(result, BenchmarkResult)
    assert result.wall_clock_sec > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmarks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.benchmarks'`

- [ ] **Step 3: Create src/benchmarks/__init__.py**

```python
"""Benchmark implementations."""
```

- [ ] **Step 4: Implement src/benchmarks/access_patterns.py**

```python
"""Access pattern benchmarks: random access, sequential scan, column subset."""

import numpy as np

from src.config import Settings
from src.metrics import BenchmarkResult, BenchmarkTimer, CpuTracker, MemoryTracker
from src.storage import get_lance_dataset, get_parquet_dataset


def bench_random_access_lance(settings: Settings) -> BenchmarkResult:
    """Benchmark: read random rows from Lance dataset."""
    ds = get_lance_dataset(settings.lance_decision_path, settings)
    n_rows = ds.count_rows()
    rng = np.random.default_rng(settings.seed)
    indices = rng.choice(n_rows, size=min(settings.random_access_sample_size, n_rows), replace=False)
    indices = sorted(indices.tolist())

    first_row_latency = None

    with BenchmarkTimer() as timer, MemoryTracker() as mem, CpuTracker() as cpu:
        # Lance supports take() for O(1) random access
        table = ds.take(indices)
        if first_row_latency is None:
            first_row_latency = timer.elapsed_sec  # approximate

    return BenchmarkResult(
        wall_clock_sec=timer.elapsed_sec,
        peak_memory_mb=mem.peak_mb,
        cpu_percent=cpu.cpu_percent,
        rows_read=table.num_rows,
        first_row_latency_sec=first_row_latency,
    )


def bench_random_access_parquet(settings: Settings) -> BenchmarkResult:
    """Benchmark: read random rows from Parquet dataset."""
    ds = get_parquet_dataset(settings.parquet_decision_path, settings)
    # Get total row count by reading metadata
    full_table = ds.to_table()
    n_rows = full_table.num_rows
    rng = np.random.default_rng(settings.seed)
    indices = rng.choice(n_rows, size=min(settings.random_access_sample_size, n_rows), replace=False)
    indices = sorted(indices.tolist())

    with BenchmarkTimer() as timer, MemoryTracker() as mem, CpuTracker() as cpu:
        # Parquet: must read then take — no native random access
        table = ds.to_table()
        result_table = table.take(indices)

    return BenchmarkResult(
        wall_clock_sec=timer.elapsed_sec,
        peak_memory_mb=mem.peak_mb,
        cpu_percent=cpu.cpu_percent,
        rows_read=result_table.num_rows,
        first_row_latency_sec=None,
    )


def bench_sequential_scan_lance(settings: Settings) -> BenchmarkResult:
    """Benchmark: full table scan from Lance dataset."""
    ds = get_lance_dataset(settings.lance_decision_path, settings)

    with BenchmarkTimer() as timer, MemoryTracker() as mem, CpuTracker() as cpu:
        table = ds.to_table()

    rows = table.num_rows
    nbytes = table.nbytes
    throughput_rows = rows / timer.elapsed_sec if timer.elapsed_sec > 0 else 0
    throughput_mb = (nbytes / (1024 * 1024)) / timer.elapsed_sec if timer.elapsed_sec > 0 else 0

    return BenchmarkResult(
        wall_clock_sec=timer.elapsed_sec,
        peak_memory_mb=mem.peak_mb,
        cpu_percent=cpu.cpu_percent,
        rows_read=rows,
        bytes_read=nbytes,
        throughput_rows_per_sec=throughput_rows,
        throughput_mb_per_sec=throughput_mb,
    )


def bench_sequential_scan_parquet(settings: Settings) -> BenchmarkResult:
    """Benchmark: full table scan from Parquet dataset."""
    ds = get_parquet_dataset(settings.parquet_decision_path, settings)

    with BenchmarkTimer() as timer, MemoryTracker() as mem, CpuTracker() as cpu:
        table = ds.to_table()

    rows = table.num_rows
    nbytes = table.nbytes
    throughput_rows = rows / timer.elapsed_sec if timer.elapsed_sec > 0 else 0
    throughput_mb = (nbytes / (1024 * 1024)) / timer.elapsed_sec if timer.elapsed_sec > 0 else 0

    return BenchmarkResult(
        wall_clock_sec=timer.elapsed_sec,
        peak_memory_mb=mem.peak_mb,
        cpu_percent=cpu.cpu_percent,
        rows_read=rows,
        bytes_read=nbytes,
        throughput_rows_per_sec=throughput_rows,
        throughput_mb_per_sec=throughput_mb,
    )


def bench_column_subset_lance(settings: Settings) -> BenchmarkResult:
    """Benchmark: read subset of columns from Lance dataset."""
    ds = get_lance_dataset(settings.lance_decision_path, settings)
    columns = list(settings.column_subset)

    with BenchmarkTimer() as timer, MemoryTracker() as mem, CpuTracker() as cpu:
        table = ds.to_table(columns=columns)

    rows = table.num_rows
    nbytes = table.nbytes
    throughput_rows = rows / timer.elapsed_sec if timer.elapsed_sec > 0 else 0
    throughput_mb = (nbytes / (1024 * 1024)) / timer.elapsed_sec if timer.elapsed_sec > 0 else 0

    return BenchmarkResult(
        wall_clock_sec=timer.elapsed_sec,
        peak_memory_mb=mem.peak_mb,
        cpu_percent=cpu.cpu_percent,
        rows_read=rows,
        bytes_read=nbytes,
        throughput_rows_per_sec=throughput_rows,
        throughput_mb_per_sec=throughput_mb,
    )


def bench_column_subset_parquet(settings: Settings) -> BenchmarkResult:
    """Benchmark: read subset of columns from Parquet dataset."""
    ds = get_parquet_dataset(settings.parquet_decision_path, settings)
    columns = list(settings.column_subset)

    with BenchmarkTimer() as timer, MemoryTracker() as mem, CpuTracker() as cpu:
        table = ds.to_table(columns=columns)

    rows = table.num_rows
    nbytes = table.nbytes
    throughput_rows = rows / timer.elapsed_sec if timer.elapsed_sec > 0 else 0
    throughput_mb = (nbytes / (1024 * 1024)) / timer.elapsed_sec if timer.elapsed_sec > 0 else 0

    return BenchmarkResult(
        wall_clock_sec=timer.elapsed_sec,
        peak_memory_mb=mem.peak_mb,
        cpu_percent=cpu.cpu_percent,
        rows_read=rows,
        bytes_read=nbytes,
        throughput_rows_per_sec=throughput_rows,
        throughput_mb_per_sec=throughput_mb,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmarks.py -v -m integration`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/benchmarks/__init__.py src/benchmarks/access_patterns.py tests/test_benchmarks.py
git commit -m "feat: access pattern benchmarks — random, sequential, column subset"
```

---

### Task 7: Model Wrappers & Training Benchmarks

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/decision_xgb.py`
- Create: `src/models/volume_mlp.py`
- Create: `src/benchmarks/training.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: `src.config.Settings`, `src.storage.get_lance_dataset`, `src.storage.get_parquet_dataset`, `src.metrics.BenchmarkResult`, `src.metrics.BenchmarkTimer`, `src.metrics.MemoryTracker`, `src.metrics.CpuTracker`
- Produces:
  - `src.models.decision_xgb.train_xgboost(X: np.ndarray, y: np.ndarray) -> xgb.Booster`
  - `src.models.volume_mlp.VolumeMLP` (nn.Module), `train_mlp(dataloader, epochs=5) -> VolumeMLP`
  - `bench_xgboost_lance(settings: Settings) -> dict[str, BenchmarkResult]` (keys: "data_load", "to_dmatrix", "train", "total")
  - `bench_xgboost_parquet(settings: Settings) -> dict[str, BenchmarkResult]`
  - `bench_pytorch_lance(settings: Settings) -> dict[str, BenchmarkResult]` (keys: "data_load", "dataloader_init", "train_5_epochs", "total")
  - `bench_pytorch_parquet(settings: Settings) -> dict[str, BenchmarkResult]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_training.py`:

```python
"""Training benchmark tests — small scale, require MinIO."""

import pytest

from src.config import Settings
from src.metrics import BenchmarkResult
from src.storage import ensure_bucket, write_lance_to_minio, write_parquet_to_minio

TRAIN_SETTINGS = Settings(
    num_sites=3,
    num_tanks_per_site=2,
    num_days=5,
    lance_decision_path="s3://benchmark/test-train/lance/decision",
    parquet_decision_path="s3://benchmark/test-train/parquet/decision",
    lance_volume_path="s3://benchmark/test-train/lance/volume",
    parquet_volume_path="s3://benchmark/test-train/parquet/volume",
)


@pytest.fixture(scope="module", autouse=True)
def setup_training_data():
    """Write small test datasets for training benchmarks."""
    from src.generate import generate_decision_features, generate_volume_events

    ensure_bucket(TRAIN_SETTINGS)
    decision_table = generate_decision_features(TRAIN_SETTINGS)
    write_lance_to_minio(decision_table, TRAIN_SETTINGS.lance_decision_path, TRAIN_SETTINGS)
    write_parquet_to_minio(
        decision_table, TRAIN_SETTINGS.parquet_decision_path, TRAIN_SETTINGS, partition_col="feature_date"
    )
    volume_table = generate_volume_events(TRAIN_SETTINGS)
    write_lance_to_minio(volume_table, TRAIN_SETTINGS.lance_volume_path, TRAIN_SETTINGS)
    write_parquet_to_minio(
        volume_table, TRAIN_SETTINGS.parquet_volume_path, TRAIN_SETTINGS, partition_col=None
    )


@pytest.mark.integration
def test_xgboost_lance():
    from src.benchmarks.training import bench_xgboost_lance

    results = bench_xgboost_lance(TRAIN_SETTINGS)
    assert "data_load" in results
    assert "to_dmatrix" in results
    assert "train" in results
    assert "total" in results
    assert results["total"].wall_clock_sec > 0


@pytest.mark.integration
def test_xgboost_parquet():
    from src.benchmarks.training import bench_xgboost_parquet

    results = bench_xgboost_parquet(TRAIN_SETTINGS)
    assert "total" in results
    assert results["total"].wall_clock_sec > 0


@pytest.mark.integration
def test_pytorch_lance():
    from src.benchmarks.training import bench_pytorch_lance

    results = bench_pytorch_lance(TRAIN_SETTINGS)
    assert "data_load" in results
    assert "dataloader_init" in results
    assert "train_5_epochs" in results
    assert "total" in results
    assert results["total"].wall_clock_sec > 0


@pytest.mark.integration
def test_pytorch_parquet():
    from src.benchmarks.training import bench_pytorch_parquet

    results = bench_pytorch_parquet(TRAIN_SETTINGS)
    assert "total" in results
    assert results["total"].wall_clock_sec > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 3: Create src/models/__init__.py**

```python
"""ML model wrappers for benchmarking."""
```

- [ ] **Step 4: Implement src/models/decision_xgb.py**

```python
"""XGBoost Decision Model — binary classification (deliver today?)."""

import numpy as np
import xgboost as xgb

# Fixed params — we're benchmarking I/O, not model quality
XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.1,
    "eval_metric": "logloss",
    "nthread": 4,
    "verbosity": 0,
}
NUM_BOOST_ROUND = 100


def train_xgboost(X: np.ndarray, y: np.ndarray) -> xgb.Booster:
    """Train an XGBoost classifier with fixed hyperparameters.

    Args:
        X: Feature matrix (n_samples, n_features), float64.
        y: Target array (n_samples,), binary 0/1.

    Returns:
        Trained xgb.Booster.
    """
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    return booster
```

- [ ] **Step 5: Implement src/models/volume_mlp.py**

```python
"""PyTorch MLP Volume Model — regression (predict delivery volume)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class VolumeMLP(nn.Module):
    """Simple 2-layer MLP for volume prediction."""

    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    dataloader: DataLoader,
    input_dim: int = 5,
    epochs: int = 5,
    lr: float = 1e-3,
) -> VolumeMLP:
    """Train the MLP model for a fixed number of epochs.

    Args:
        dataloader: PyTorch DataLoader yielding (features, targets) batches.
        input_dim: Number of input features.
        epochs: Number of training epochs.
        lr: Learning rate.

    Returns:
        Trained VolumeMLP model.
    """
    device = torch.device("cpu")  # CPU only for this benchmark
    model = VolumeMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X).squeeze()
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

    return model


def create_dataloader_from_numpy(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1024,
    shuffle: bool = True,
) -> DataLoader:
    """Create a standard PyTorch DataLoader from numpy arrays.

    This is the Parquet approach: load everything to memory, wrap in TensorDataset.
    """
    X_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32))
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
```

- [ ] **Step 6: Implement src/benchmarks/training.py**

```python
"""Training benchmarks: XGBoost + PyTorch end-to-end with step timing."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import Settings
from src.metrics import BenchmarkResult, BenchmarkTimer, CpuTracker, MemoryTracker
from src.models.decision_xgb import train_xgboost
from src.models.volume_mlp import VolumeMLP, create_dataloader_from_numpy, train_mlp
from src.storage import get_lance_dataset, get_parquet_dataset

# Feature columns for XGBoost (all numeric columns except target)
DECISION_FEATURE_COLS = [
    "tank_capacity", "open_inventory", "avg_sale_7d", "avg_sale_30d",
    "overdue_ratio", "inv_days_cover", "dow_hist_rate", "hist_rate",
    "delivery_sum_28d", "proj_end_fill_ratio", "group_overdue_pct",
    "group_cover_pct", "last_delivery_days_ago", "intransit_volume",
    "usage_day_at_approval", "current_inventory_at_approval",
    "day_of_week", "day_of_month", "month",
    "delivery_volume_lag1", "delivery_volume_lag7",
    "rolling_delivery_count_7d", "rolling_delivery_count_30d",
    "pair_deliv_mean", "yoy_deliv",
]
DECISION_TARGET_COL = "delivery_occurred"

# Feature columns for Volume MLP
VOLUME_FEATURE_COLS = ["reading_hour", "atg_start"]
VOLUME_TARGET_COL = "atg_diff"


def bench_xgboost_lance(settings: Settings) -> dict[str, BenchmarkResult]:
    """Benchmark XGBoost training with Lance data source."""
    results = {}

    # Step 1: Data load
    with BenchmarkTimer() as t1, MemoryTracker() as m1, CpuTracker() as c1:
        ds = get_lance_dataset(settings.lance_decision_path, settings)
        table = ds.to_table(columns=DECISION_FEATURE_COLS + [DECISION_TARGET_COL])

    results["data_load"] = BenchmarkResult(
        wall_clock_sec=t1.elapsed_sec, peak_memory_mb=m1.peak_mb, cpu_percent=c1.cpu_percent,
        rows_read=table.num_rows,
    )

    # Step 2: Convert to DMatrix
    with BenchmarkTimer() as t2, MemoryTracker() as m2, CpuTracker() as c2:
        X = table.select(DECISION_FEATURE_COLS).to_pandas().to_numpy(dtype=np.float64)
        y = table.column(DECISION_TARGET_COL).to_numpy().astype(np.float64)

    results["to_dmatrix"] = BenchmarkResult(
        wall_clock_sec=t2.elapsed_sec, peak_memory_mb=m2.peak_mb, cpu_percent=c2.cpu_percent,
    )

    # Step 3: Train
    with BenchmarkTimer() as t3, MemoryTracker() as m3, CpuTracker() as c3:
        train_xgboost(X, y)

    results["train"] = BenchmarkResult(
        wall_clock_sec=t3.elapsed_sec, peak_memory_mb=m3.peak_mb, cpu_percent=c3.cpu_percent,
    )

    # Total
    total_time = t1.elapsed_sec + t2.elapsed_sec + t3.elapsed_sec
    total_mem = max(m1.peak_mb, m2.peak_mb, m3.peak_mb)
    results["total"] = BenchmarkResult(
        wall_clock_sec=total_time, peak_memory_mb=total_mem,
    )

    return results


def bench_xgboost_parquet(settings: Settings) -> dict[str, BenchmarkResult]:
    """Benchmark XGBoost training with Parquet data source."""
    results = {}

    # Step 1: Data load
    with BenchmarkTimer() as t1, MemoryTracker() as m1, CpuTracker() as c1:
        ds = get_parquet_dataset(settings.parquet_decision_path, settings)
        table = ds.to_table(columns=DECISION_FEATURE_COLS + [DECISION_TARGET_COL])

    results["data_load"] = BenchmarkResult(
        wall_clock_sec=t1.elapsed_sec, peak_memory_mb=m1.peak_mb, cpu_percent=c1.cpu_percent,
        rows_read=table.num_rows,
    )

    # Step 2: Convert to DMatrix
    with BenchmarkTimer() as t2, MemoryTracker() as m2, CpuTracker() as c2:
        X = table.select(DECISION_FEATURE_COLS).to_pandas().to_numpy(dtype=np.float64)
        y = table.column(DECISION_TARGET_COL).to_numpy().astype(np.float64)

    results["to_dmatrix"] = BenchmarkResult(
        wall_clock_sec=t2.elapsed_sec, peak_memory_mb=m2.peak_mb, cpu_percent=c2.cpu_percent,
    )

    # Step 3: Train
    with BenchmarkTimer() as t3, MemoryTracker() as m3, CpuTracker() as c3:
        train_xgboost(X, y)

    results["train"] = BenchmarkResult(
        wall_clock_sec=t3.elapsed_sec, peak_memory_mb=m3.peak_mb, cpu_percent=c3.cpu_percent,
    )

    # Total
    total_time = t1.elapsed_sec + t2.elapsed_sec + t3.elapsed_sec
    total_mem = max(m1.peak_mb, m2.peak_mb, m3.peak_mb)
    results["total"] = BenchmarkResult(
        wall_clock_sec=total_time, peak_memory_mb=total_mem,
    )

    return results


def bench_pytorch_lance(settings: Settings) -> dict[str, BenchmarkResult]:
    """Benchmark PyTorch MLP training with Lance data source.

    Uses lance's native dataset for efficient random access during training.
    """
    results = {}

    # Step 1: Data load — load full table from Lance
    # Note: In production, you'd use lance.torch.data.LanceDataset for true
    # streaming. Here we compare the two loading strategies.
    with BenchmarkTimer() as t1, MemoryTracker() as m1, CpuTracker() as c1:
        ds = get_lance_dataset(settings.lance_volume_path, settings)
        table = ds.to_table(columns=VOLUME_FEATURE_COLS + [VOLUME_TARGET_COL])

    results["data_load"] = BenchmarkResult(
        wall_clock_sec=t1.elapsed_sec, peak_memory_mb=m1.peak_mb, cpu_percent=c1.cpu_percent,
        rows_read=table.num_rows,
    )

    # Step 2: DataLoader init
    with BenchmarkTimer() as t2, MemoryTracker() as m2, CpuTracker() as c2:
        X = table.select(VOLUME_FEATURE_COLS).to_pandas().to_numpy(dtype=np.float32)
        y = table.column(VOLUME_TARGET_COL).to_numpy().astype(np.float32)
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    results["dataloader_init"] = BenchmarkResult(
        wall_clock_sec=t2.elapsed_sec, peak_memory_mb=m2.peak_mb, cpu_percent=c2.cpu_percent,
    )

    # Step 3: Train 5 epochs
    with BenchmarkTimer() as t3, MemoryTracker() as m3, CpuTracker() as c3:
        train_mlp(dataloader, input_dim=len(VOLUME_FEATURE_COLS), epochs=5)

    results["train_5_epochs"] = BenchmarkResult(
        wall_clock_sec=t3.elapsed_sec, peak_memory_mb=m3.peak_mb, cpu_percent=c3.cpu_percent,
    )

    # Total
    total_time = t1.elapsed_sec + t2.elapsed_sec + t3.elapsed_sec
    total_mem = max(m1.peak_mb, m2.peak_mb, m3.peak_mb)
    results["total"] = BenchmarkResult(
        wall_clock_sec=total_time, peak_memory_mb=total_mem,
    )

    return results


def bench_pytorch_parquet(settings: Settings) -> dict[str, BenchmarkResult]:
    """Benchmark PyTorch MLP training with Parquet data source.

    Must load entire dataset to memory before training.
    """
    results = {}

    # Step 1: Data load
    with BenchmarkTimer() as t1, MemoryTracker() as m1, CpuTracker() as c1:
        ds = get_parquet_dataset(settings.parquet_volume_path, settings)
        table = ds.to_table(columns=VOLUME_FEATURE_COLS + [VOLUME_TARGET_COL])

    results["data_load"] = BenchmarkResult(
        wall_clock_sec=t1.elapsed_sec, peak_memory_mb=m1.peak_mb, cpu_percent=c1.cpu_percent,
        rows_read=table.num_rows,
    )

    # Step 2: DataLoader init
    with BenchmarkTimer() as t2, MemoryTracker() as m2, CpuTracker() as c2:
        X = table.select(VOLUME_FEATURE_COLS).to_pandas().to_numpy(dtype=np.float32)
        y = table.column(VOLUME_TARGET_COL).to_numpy().astype(np.float32)
        dataloader = create_dataloader_from_numpy(X, y, batch_size=1024, shuffle=True)

    results["dataloader_init"] = BenchmarkResult(
        wall_clock_sec=t2.elapsed_sec, peak_memory_mb=m2.peak_mb, cpu_percent=c2.cpu_percent,
    )

    # Step 3: Train 5 epochs
    with BenchmarkTimer() as t3, MemoryTracker() as m3, CpuTracker() as c3:
        train_mlp(dataloader, input_dim=len(VOLUME_FEATURE_COLS), epochs=5)

    results["train_5_epochs"] = BenchmarkResult(
        wall_clock_sec=t3.elapsed_sec, peak_memory_mb=m3.peak_mb, cpu_percent=c3.cpu_percent,
    )

    # Total
    total_time = t1.elapsed_sec + t2.elapsed_sec + t3.elapsed_sec
    total_mem = max(m1.peak_mb, m2.peak_mb, m3.peak_mb)
    results["total"] = BenchmarkResult(
        wall_clock_sec=total_time, peak_memory_mb=total_mem,
    )

    return results
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_training.py -v -m integration`
Expected: All 4 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/models/__init__.py src/models/decision_xgb.py src/models/volume_mlp.py src/benchmarks/training.py tests/test_training.py
git commit -m "feat: model wrappers + training benchmarks — XGBoost and PyTorch"
```

---

### Task 8: HTML Report Generation

**Files:**
- Create: `src/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `src.metrics.BenchmarkResult`, results dict from benchmarks
- Produces:
  - `generate_report(results: dict, output_path: str = "report.html") -> None`: writes self-contained HTML file
  - `save_results_json(results: dict, output_path: str = "results.json") -> None`: writes raw metrics as JSON

- [ ] **Step 1: Write failing tests**

Create `tests/test_report.py`:

```python
"""Report generation tests."""

import json
import os
import tempfile

from src.metrics import BenchmarkResult
from src.report import generate_report, save_results_json


def _mock_results():
    """Create mock benchmark results for testing."""
    return {
        "environment": {
            "python_version": "3.11.0",
            "cpu": "Apple M1",
            "ram_gb": 16,
            "os": "macOS",
        },
        "file_sizes": {
            "lance_decision_mb": 45.2,
            "parquet_decision_mb": 38.1,
            "lance_volume_mb": 120.5,
            "parquet_volume_mb": 105.3,
        },
        "access_patterns": {
            "random_access": {
                "lance": {"wall_clock_mean": 0.15, "wall_clock_std": 0.02, "peak_memory_mb": 50.0, "cpu_percent": 45.0},
                "parquet": {"wall_clock_mean": 2.5, "wall_clock_std": 0.3, "peak_memory_mb": 200.0, "cpu_percent": 60.0},
            },
            "sequential_scan": {
                "lance": {"wall_clock_mean": 1.2, "wall_clock_std": 0.1, "peak_memory_mb": 300.0, "cpu_percent": 70.0},
                "parquet": {"wall_clock_mean": 1.1, "wall_clock_std": 0.1, "peak_memory_mb": 280.0, "cpu_percent": 65.0},
            },
            "column_subset": {
                "lance": {"wall_clock_mean": 0.5, "wall_clock_std": 0.05, "peak_memory_mb": 80.0, "cpu_percent": 40.0},
                "parquet": {"wall_clock_mean": 0.6, "wall_clock_std": 0.06, "peak_memory_mb": 90.0, "cpu_percent": 42.0},
            },
        },
        "training": {
            "xgboost": {
                "lance": {"data_load": 0.8, "to_dmatrix": 0.3, "train": 5.0, "total": 6.1},
                "parquet": {"data_load": 1.2, "to_dmatrix": 0.3, "train": 5.0, "total": 6.5},
            },
            "pytorch": {
                "lance": {"data_load": 0.5, "dataloader_init": 0.2, "train_5_epochs": 12.0, "total": 12.7},
                "parquet": {"data_load": 4.5, "dataloader_init": 1.5, "train_5_epochs": 12.0, "total": 18.0},
            },
        },
    }


def test_generate_report_creates_file():
    results = _mock_results()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        generate_report(results, output_path=path)
        assert os.path.exists(path)
        content = open(path).read()
        assert "<html" in content
        assert "Lance" in content
        assert "Parquet" in content
        assert "plotly" in content.lower() or "Plotly" in content
    finally:
        os.unlink(path)


def test_generate_report_contains_sections():
    results = _mock_results()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        generate_report(results, output_path=path)
        content = open(path).read()
        assert "Summary" in content
        assert "Random" in content or "random" in content
        assert "Sequential" in content or "sequential" in content
        assert "XGBoost" in content or "xgboost" in content
        assert "PyTorch" in content or "pytorch" in content
    finally:
        os.unlink(path)


def test_save_results_json():
    results = _mock_results()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        save_results_json(results, output_path=path)
        assert os.path.exists(path)
        loaded = json.loads(open(path).read())
        assert "environment" in loaded
        assert "access_patterns" in loaded
        assert "training" in loaded
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.report'`

- [ ] **Step 3: Implement src/report.py**

```python
"""HTML report generation with Plotly charts."""

import json
import platform
import sys
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _summary_card(results: dict) -> str:
    """Generate the summary card HTML."""
    ap = results.get("access_patterns", {})
    training = results.get("training", {})

    # Random access speedup
    ra = ap.get("random_access", {})
    lance_ra = ra.get("lance", {}).get("wall_clock_mean", 1)
    parquet_ra = ra.get("parquet", {}).get("wall_clock_mean", 1)
    ra_speedup = parquet_ra / lance_ra if lance_ra > 0 else 0

    # PyTorch memory difference
    pt = training.get("pytorch", {})
    lance_pt_total = pt.get("lance", {}).get("total", 0)
    parquet_pt_total = pt.get("parquet", {}).get("total", 0)
    pt_speedup = parquet_pt_total / lance_pt_total if lance_pt_total > 0 else 0

    return f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 30px; border-radius: 12px; margin-bottom: 30px; color: white;">
        <h2 style="margin-top: 0;">Summary</h2>
        <div style="display: flex; gap: 40px; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 48px; font-weight: bold; color: #4ecdc4;">{ra_speedup:.1f}x</div>
                <div>Lance faster at random access</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 48px; font-weight: bold; color: #4ecdc4;">{pt_speedup:.1f}x</div>
                <div>Lance faster for PyTorch training (total)</div>
            </div>
        </div>
    </div>
    """


def _access_pattern_chart(results: dict) -> str:
    """Generate grouped bar chart for access patterns."""
    ap = results.get("access_patterns", {})
    patterns = ["random_access", "sequential_scan", "column_subset"]
    labels = ["Random Access", "Sequential Scan", "Column Subset"]

    lance_times = [ap.get(p, {}).get("lance", {}).get("wall_clock_mean", 0) for p in patterns]
    parquet_times = [ap.get(p, {}).get("parquet", {}).get("wall_clock_mean", 0) for p in patterns]

    fig = go.Figure(data=[
        go.Bar(name="Lance", x=labels, y=lance_times, marker_color="#4ecdc4"),
        go.Bar(name="Parquet", x=labels, y=parquet_times, marker_color="#ff6b6b"),
    ])
    fig.update_layout(
        title="Access Pattern: Wall-Clock Time (lower is better)",
        yaxis_title="Seconds",
        barmode="group",
        template="plotly_dark",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _memory_chart(results: dict) -> str:
    """Generate memory comparison bar chart."""
    ap = results.get("access_patterns", {})
    patterns = ["random_access", "sequential_scan", "column_subset"]
    labels = ["Random Access", "Sequential Scan", "Column Subset"]

    lance_mem = [ap.get(p, {}).get("lance", {}).get("peak_memory_mb", 0) for p in patterns]
    parquet_mem = [ap.get(p, {}).get("parquet", {}).get("peak_memory_mb", 0) for p in patterns]

    fig = go.Figure(data=[
        go.Bar(name="Lance", x=labels, y=lance_mem, marker_color="#4ecdc4"),
        go.Bar(name="Parquet", x=labels, y=parquet_mem, marker_color="#ff6b6b"),
    ])
    fig.update_layout(
        title="Peak Memory Usage (lower is better)",
        yaxis_title="MB",
        barmode="group",
        template="plotly_dark",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _cpu_chart(results: dict) -> str:
    """Generate CPU utilization bar chart."""
    ap = results.get("access_patterns", {})
    patterns = ["random_access", "sequential_scan", "column_subset"]
    labels = ["Random Access", "Sequential Scan", "Column Subset"]

    lance_cpu = [ap.get(p, {}).get("lance", {}).get("cpu_percent", 0) for p in patterns]
    parquet_cpu = [ap.get(p, {}).get("parquet", {}).get("cpu_percent", 0) for p in patterns]

    fig = go.Figure(data=[
        go.Bar(name="Lance", x=labels, y=lance_cpu, marker_color="#4ecdc4"),
        go.Bar(name="Parquet", x=labels, y=parquet_cpu, marker_color="#ff6b6b"),
    ])
    fig.update_layout(
        title="CPU Utilization (%)",
        yaxis_title="%",
        barmode="group",
        template="plotly_dark",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _training_breakdown_chart(results: dict) -> str:
    """Generate stacked bar chart for training step breakdown."""
    training = results.get("training", {})

    # XGBoost steps
    xgb_lance = training.get("xgboost", {}).get("lance", {})
    xgb_parquet = training.get("xgboost", {}).get("parquet", {})

    # PyTorch steps
    pt_lance = training.get("pytorch", {}).get("lance", {})
    pt_parquet = training.get("pytorch", {}).get("parquet", {})

    categories = ["XGBoost (Lance)", "XGBoost (Parquet)", "PyTorch (Lance)", "PyTorch (Parquet)"]

    # For XGBoost: data_load, to_dmatrix, train
    # For PyTorch: data_load, dataloader_init, train_5_epochs
    data_load = [
        xgb_lance.get("data_load", 0), xgb_parquet.get("data_load", 0),
        pt_lance.get("data_load", 0), pt_parquet.get("data_load", 0),
    ]
    convert = [
        xgb_lance.get("to_dmatrix", 0), xgb_parquet.get("to_dmatrix", 0),
        pt_lance.get("dataloader_init", 0), pt_parquet.get("dataloader_init", 0),
    ]
    train = [
        xgb_lance.get("train", 0), xgb_parquet.get("train", 0),
        pt_lance.get("train_5_epochs", 0), pt_parquet.get("train_5_epochs", 0),
    ]

    fig = go.Figure(data=[
        go.Bar(name="Data Load", x=categories, y=data_load, marker_color="#4ecdc4"),
        go.Bar(name="Convert/Init", x=categories, y=convert, marker_color="#45b7d1"),
        go.Bar(name="Train", x=categories, y=train, marker_color="#96ceb4"),
    ])
    fig.update_layout(
        title="Training Step Breakdown (seconds)",
        yaxis_title="Seconds",
        barmode="stack",
        template="plotly_dark",
        height=400,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _storage_table(results: dict) -> str:
    """Generate HTML table for storage efficiency."""
    fs = results.get("file_sizes", {})
    rows = ""
    for key, val in fs.items():
        label = key.replace("_mb", " (MB)").replace("_", " ").title()
        rows += f"<tr><td>{label}</td><td>{val:.2f}</td></tr>"

    return f"""
    <div style="margin: 20px 0;">
        <h3>Storage Efficiency</h3>
        <table style="width: 100%; border-collapse: collapse; background: #1e1e2e; color: white;">
            <thead><tr style="border-bottom: 2px solid #4ecdc4;">
                <th style="padding: 10px; text-align: left;">Dataset</th>
                <th style="padding: 10px; text-align: right;">Size (MB)</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """


def _environment_section(results: dict) -> str:
    """Generate environment info section."""
    env = results.get("environment", {})
    items = "".join(f"<li><strong>{k}:</strong> {v}</li>" for k, v in env.items())
    return f"""
    <div style="margin: 20px 0; padding: 20px; background: #1e1e2e; border-radius: 8px; color: #ccc;">
        <h3>Environment</h3>
        <ul>{items}</ul>
    </div>
    """


def generate_report(results: dict, output_path: str = "report.html") -> None:
    """Generate a self-contained HTML report with Plotly charts.

    Args:
        results: Full benchmark results dict.
        output_path: Path to write the HTML file.
    """
    import plotly

    plotly_js = f'<script src="https://cdn.plot.ly/plotly-{plotly.__version__}.min.js"></script>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lance vs Parquet Benchmark Report</title>
    {plotly_js}
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #e6e6e6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        h1 {{ color: #4ecdc4; border-bottom: 2px solid #4ecdc4; padding-bottom: 10px; }}
        h2 {{ color: #ccc; margin-top: 40px; }}
        .chart-container {{ margin: 30px 0; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Lance vs Parquet Benchmark Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    {_summary_card(results)}

    <h2>Access Pattern Benchmarks</h2>
    <div class="chart-container">{_access_pattern_chart(results)}</div>

    <h2>Memory Usage</h2>
    <div class="chart-container">{_memory_chart(results)}</div>

    <h2>CPU Utilization</h2>
    <div class="chart-container">{_cpu_chart(results)}</div>

    <h2>Training Step Breakdown</h2>
    <div class="chart-container">{_training_breakdown_chart(results)}</div>

    {_storage_table(results)}
    {_environment_section(results)}
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)


def save_results_json(results: dict, output_path: str = "results.json") -> None:
    """Save raw benchmark results as JSON.

    Args:
        results: Full benchmark results dict.
        output_path: Path to write the JSON file.
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/report.py tests/test_report.py
git commit -m "feat: HTML report generation with Plotly charts"
```

---

### Task 9: Main Orchestrator (benchmark.py)

**Files:**
- Create: `src/benchmark.py`
- Create: `src/__main__.py`

**Interfaces:**
- Consumes: All previous modules — `config.Settings`, `generate.*`, `storage.*`, `metrics.run_benchmark`, `benchmarks.access_patterns.*`, `benchmarks.training.*`, `report.*`
- Produces: `main() -> None` — the single entry point that generates data, runs all benchmarks, and produces the report

- [ ] **Step 1: Implement src/benchmark.py**

```python
"""Main benchmark orchestrator — generates data, runs benchmarks, produces report."""

import platform
import statistics
import sys
import time

import numpy as np
import psutil

from src.benchmarks.access_patterns import (
    bench_column_subset_lance,
    bench_column_subset_parquet,
    bench_random_access_lance,
    bench_random_access_parquet,
    bench_sequential_scan_lance,
    bench_sequential_scan_parquet,
)
from src.benchmarks.training import (
    bench_pytorch_lance,
    bench_pytorch_parquet,
    bench_xgboost_lance,
    bench_xgboost_parquet,
)
from src.config import Settings
from src.generate import generate_decision_features, generate_volume_events
from src.metrics import BenchmarkResult, run_benchmark
from src.report import generate_report, save_results_json
from src.storage import (
    ensure_bucket,
    get_dataset_size_mb,
    write_lance_to_minio,
    write_parquet_to_minio,
)


def _get_environment_info() -> dict:
    """Collect environment information for the report."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or "Unknown",
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "os": platform.system(),
    }


def _summarize_results(results: list[BenchmarkResult]) -> dict:
    """Summarize a list of BenchmarkResults into mean/std dict."""
    times = [r.wall_clock_sec for r in results]
    mems = [r.peak_memory_mb for r in results]
    cpus = [r.cpu_percent for r in results if r.cpu_percent is not None]

    summary = {
        "wall_clock_mean": statistics.mean(times),
        "wall_clock_std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "peak_memory_mb": max(mems),
        "cpu_percent": statistics.mean(cpus) if cpus else 0.0,
    }

    # Add throughput if available
    throughputs = [r.throughput_rows_per_sec for r in results if r.throughput_rows_per_sec]
    if throughputs:
        summary["throughput_rows_per_sec"] = statistics.mean(throughputs)

    throughputs_mb = [r.throughput_mb_per_sec for r in results if r.throughput_mb_per_sec]
    if throughputs_mb:
        summary["throughput_mb_per_sec"] = statistics.mean(throughputs_mb)

    # Add first-row latency if available
    latencies = [r.first_row_latency_sec for r in results if r.first_row_latency_sec]
    if latencies:
        summary["first_row_latency_sec"] = statistics.mean(latencies)

    # Add S3 calls if available
    s3_calls = [r.s3_calls for r in results if r.s3_calls is not None]
    if s3_calls:
        summary["s3_calls_mean"] = statistics.mean(s3_calls)

    return summary


def _summarize_training_results(results_list: list[dict[str, BenchmarkResult]]) -> dict:
    """Summarize training benchmark results (step-level timing)."""
    # Collect step times across repetitions
    steps = results_list[0].keys()
    summary = {}
    for step in steps:
        times = [r[step].wall_clock_sec for r in results_list]
        summary[step] = statistics.mean(times)
    return summary


def _run_training_benchmark(
    fn: callable,
    repeats: int = 3,
    warmup: int = 1,
) -> list[dict[str, BenchmarkResult]]:
    """Run a training benchmark function that returns dict[str, BenchmarkResult].

    Similar to run_benchmark but for functions returning step-level dicts.
    """
    import gc

    # Warmup
    for _ in range(warmup):
        fn()
        gc.collect()

    # Measured runs
    results = []
    for _ in range(repeats):
        result = fn()
        results.append(result)
        gc.collect()

    return results


def main() -> None:
    """Run the full benchmark pipeline."""
    settings = Settings()
    print("=" * 60)
    print("Lance vs Parquet Benchmark")
    print("=" * 60)

    # Step 1: Setup
    print("\n[1/6] Setting up MinIO bucket...")
    ensure_bucket(settings)

    # Step 2: Generate and write data
    print("\n[2/6] Generating synthetic data...")
    print(f"  Decision Model: {settings.total_decision_rows:,} rows, 32 columns")
    print(f"  Volume Model:   {settings.total_volume_rows:,} rows, 6 columns")

    decision_table = generate_decision_features(settings)
    volume_table = generate_volume_events(settings)

    print("  Writing Parquet to MinIO...")
    write_parquet_to_minio(
        decision_table, settings.parquet_decision_path, settings, partition_col="feature_date"
    )
    write_parquet_to_minio(
        volume_table, settings.parquet_volume_path, settings, partition_col=None
    )

    print("  Writing Lance to MinIO...")
    write_lance_to_minio(decision_table, settings.lance_decision_path, settings)
    write_lance_to_minio(volume_table, settings.lance_volume_path, settings)

    # Get file sizes
    file_sizes = {
        "lance_decision_mb": get_dataset_size_mb(settings.lance_decision_path, settings),
        "parquet_decision_mb": get_dataset_size_mb(settings.parquet_decision_path, settings),
        "lance_volume_mb": get_dataset_size_mb(settings.lance_volume_path, settings),
        "parquet_volume_mb": get_dataset_size_mb(settings.parquet_volume_path, settings),
    }
    print(f"  File sizes: Lance Decision={file_sizes['lance_decision_mb']:.1f}MB, "
          f"Parquet Decision={file_sizes['parquet_decision_mb']:.1f}MB")

    # Step 3: Access pattern benchmarks
    print(f"\n[3/6] Running access pattern benchmarks "
          f"({settings.warmup_runs} warmup + {settings.benchmark_repeats} measured)...")

    print("  Random access (Lance)...")
    lance_ra = run_benchmark(
        lambda: bench_random_access_lance(settings),
        repeats=settings.benchmark_repeats, warmup=settings.warmup_runs,
    )
    print("  Random access (Parquet)...")
    parquet_ra = run_benchmark(
        lambda: bench_random_access_parquet(settings),
        repeats=settings.benchmark_repeats, warmup=settings.warmup_runs,
    )

    print("  Sequential scan (Lance)...")
    lance_seq = run_benchmark(
        lambda: bench_sequential_scan_lance(settings),
        repeats=settings.benchmark_repeats, warmup=settings.warmup_runs,
    )
    print("  Sequential scan (Parquet)...")
    parquet_seq = run_benchmark(
        lambda: bench_sequential_scan_parquet(settings),
        repeats=settings.benchmark_repeats, warmup=settings.warmup_runs,
    )

    print("  Column subset (Lance)...")
    lance_col = run_benchmark(
        lambda: bench_column_subset_lance(settings),
        repeats=settings.benchmark_repeats, warmup=settings.warmup_runs,
    )
    print("  Column subset (Parquet)...")
    parquet_col = run_benchmark(
        lambda: bench_column_subset_parquet(settings),
        repeats=settings.benchmark_repeats, warmup=settings.warmup_runs,
    )

    # Step 4: Training benchmarks
    print(f"\n[4/6] Running XGBoost training benchmark...")
    lance_xgb = _run_training_benchmark(
        lambda: bench_xgboost_lance(settings),
        repeats=settings.benchmark_repeats, warmup=settings.warmup_runs,
    )
    parquet_xgb = _run_training_benchmark(
        lambda: bench_xgboost_parquet(settings),
        repeats=settings.benchmark_repeats, warmup=settings.warmup_runs,
    )

    print(f"\n[5/6] Running PyTorch training benchmark...")
    lance_pt = _run_training_benchmark(
        lambda: bench_pytorch_lance(settings),
        repeats=settings.benchmark_repeats, warmup=settings.warmup_runs,
    )
    parquet_pt = _run_training_benchmark(
        lambda: bench_pytorch_parquet(settings),
        repeats=settings.benchmark_repeats, warmup=settings.warmup_runs,
    )

    # Step 5: Compile results
    print("\n[6/6] Generating report...")
    all_results = {
        "environment": _get_environment_info(),
        "file_sizes": file_sizes,
        "access_patterns": {
            "random_access": {
                "lance": _summarize_results(lance_ra),
                "parquet": _summarize_results(parquet_ra),
            },
            "sequential_scan": {
                "lance": _summarize_results(lance_seq),
                "parquet": _summarize_results(parquet_seq),
            },
            "column_subset": {
                "lance": _summarize_results(lance_col),
                "parquet": _summarize_results(parquet_col),
            },
        },
        "training": {
            "xgboost": {
                "lance": _summarize_training_results(lance_xgb),
                "parquet": _summarize_training_results(parquet_xgb),
            },
            "pytorch": {
                "lance": _summarize_training_results(lance_pt),
                "parquet": _summarize_training_results(parquet_pt),
            },
        },
    }

    # Step 6: Generate outputs
    generate_report(all_results, output_path="report.html")
    save_results_json(all_results, output_path="results.json")

    # Print summary
    ra_speedup = (
        all_results["access_patterns"]["random_access"]["parquet"]["wall_clock_mean"]
        / all_results["access_patterns"]["random_access"]["lance"]["wall_clock_mean"]
    )
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Random access: Lance {ra_speedup:.1f}x faster")
    print(f"  Report saved to: report.html")
    print(f"  Raw data saved to: results.json")
    print(f"{'=' * 60}")
```

- [ ] **Step 2: Create src/__main__.py**

```python
"""Entry point for `python -m src`."""

from src.benchmark import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the full pipeline runs end-to-end (requires MinIO)**

Run: `docker compose up -d --wait && uv run python -m src.benchmark`
Expected: Benchmark runs, produces `report.html` and `results.json` in project root.

- [ ] **Step 4: Verify report.html opens in browser**

Run: `open report.html` (macOS) or `xdg-open report.html` (Linux)
Expected: Interactive HTML report with charts showing Lance vs Parquet comparison.

- [ ] **Step 5: Commit**

```bash
git add src/benchmark.py src/__main__.py
git commit -m "feat: main orchestrator — end-to-end benchmark pipeline"
```

- [ ] **Step 6: Add report outputs to .gitignore and commit**

Create `.gitignore`:

```
report.html
results.json
__pycache__/
*.pyc
.venv/
```

```bash
git add .gitignore
git commit -m "chore: add .gitignore for generated outputs"
```

---
