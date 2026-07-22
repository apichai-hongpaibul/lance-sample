#!/usr/bin/env bash
set -euo pipefail

echo "Starting MinIO..."
docker compose up -d --wait

echo "Installing dependencies..."
uv sync

echo "Running benchmark..."
uv run python -m src.benchmark

echo "Done! Open report.html to view results."
