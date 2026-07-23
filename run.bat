@echo off
setlocal EnableDelayedExpansion

set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1

echo Starting MinIO...
docker compose up -d --wait
if %errorlevel% neq 0 exit /b %errorlevel%

echo Installing dependencies...
uv sync
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running benchmark...
uv run python -m src.benchmark
if %errorlevel% neq 0 exit /b %errorlevel%

echo Done! Open report.html to view results.
