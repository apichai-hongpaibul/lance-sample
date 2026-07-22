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
