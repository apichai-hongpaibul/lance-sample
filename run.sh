#!/usr/bin/env bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Starting MinIO..."
docker compose up -d --wait

echo "Installing dependencies..."
uv sync

echo "Running benchmark..."
uv run python -m src.benchmark

echo "Done! Open report.html to view results."
