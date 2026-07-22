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
