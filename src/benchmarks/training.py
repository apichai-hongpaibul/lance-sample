"""Training benchmarks: XGBoost + PyTorch end-to-end with step timing."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import Settings
from src.metrics import BenchmarkResult, BenchmarkTimer, CpuTracker, MemoryTracker
from src.models.decision_xgb import train_xgboost
from src.models.volume_mlp import create_dataloader_from_numpy, train_mlp
from src.storage import get_lance_dataset, get_parquet_dataset

# Feature columns for XGBoost (all numeric columns except target)
DECISION_FEATURE_COLS = [
    "tank_capacity",
    "open_inventory",
    "avg_sale_7d",
    "avg_sale_30d",
    "overdue_ratio",
    "inv_days_cover",
    "dow_hist_rate",
    "hist_rate",
    "delivery_sum_28d",
    "proj_end_fill_ratio",
    "group_overdue_pct",
    "group_cover_pct",
    "last_delivery_days_ago",
    "intransit_volume",
    "usage_day_at_approval",
    "current_inventory_at_approval",
    "day_of_week",
    "day_of_month",
    "month",
    "delivery_volume_lag1",
    "delivery_volume_lag7",
    "rolling_delivery_count_7d",
    "rolling_delivery_count_30d",
    "pair_deliv_mean",
    "yoy_deliv",
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
        wall_clock_sec=t1.elapsed_sec,
        peak_memory_mb=m1.peak_mb,
        cpu_percent=c1.cpu_percent,
        rows_read=table.num_rows,
    )

    # Step 2: Convert to DMatrix
    with BenchmarkTimer() as t2, MemoryTracker() as m2, CpuTracker() as c2:
        X = (
            table.select(DECISION_FEATURE_COLS)
            .to_pandas()
            .to_numpy(dtype=np.float64)
        )
        y = table.column(DECISION_TARGET_COL).to_numpy().astype(np.float64)

    results["to_dmatrix"] = BenchmarkResult(
        wall_clock_sec=t2.elapsed_sec,
        peak_memory_mb=m2.peak_mb,
        cpu_percent=c2.cpu_percent,
    )

    # Step 3: Train
    with BenchmarkTimer() as t3, MemoryTracker() as m3, CpuTracker() as c3:
        train_xgboost(X, y)

    results["train"] = BenchmarkResult(
        wall_clock_sec=t3.elapsed_sec,
        peak_memory_mb=m3.peak_mb,
        cpu_percent=c3.cpu_percent,
    )

    # Total
    total_time = t1.elapsed_sec + t2.elapsed_sec + t3.elapsed_sec
    total_mem = max(m1.peak_mb, m2.peak_mb, m3.peak_mb)
    results["total"] = BenchmarkResult(
        wall_clock_sec=total_time,
        peak_memory_mb=total_mem,
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
        wall_clock_sec=t1.elapsed_sec,
        peak_memory_mb=m1.peak_mb,
        cpu_percent=c1.cpu_percent,
        rows_read=table.num_rows,
    )

    # Step 2: Convert to DMatrix
    with BenchmarkTimer() as t2, MemoryTracker() as m2, CpuTracker() as c2:
        X = (
            table.select(DECISION_FEATURE_COLS)
            .to_pandas()
            .to_numpy(dtype=np.float64)
        )
        y = table.column(DECISION_TARGET_COL).to_numpy().astype(np.float64)

    results["to_dmatrix"] = BenchmarkResult(
        wall_clock_sec=t2.elapsed_sec,
        peak_memory_mb=m2.peak_mb,
        cpu_percent=c2.cpu_percent,
    )

    # Step 3: Train
    with BenchmarkTimer() as t3, MemoryTracker() as m3, CpuTracker() as c3:
        train_xgboost(X, y)

    results["train"] = BenchmarkResult(
        wall_clock_sec=t3.elapsed_sec,
        peak_memory_mb=m3.peak_mb,
        cpu_percent=c3.cpu_percent,
    )

    # Total
    total_time = t1.elapsed_sec + t2.elapsed_sec + t3.elapsed_sec
    total_mem = max(m1.peak_mb, m2.peak_mb, m3.peak_mb)
    results["total"] = BenchmarkResult(
        wall_clock_sec=total_time,
        peak_memory_mb=total_mem,
    )

    return results


def bench_pytorch_lance(settings: Settings) -> dict[str, BenchmarkResult]:
    """Benchmark PyTorch MLP training with Lance data source."""
    results = {}

    # Step 1: Data load
    with BenchmarkTimer() as t1, MemoryTracker() as m1, CpuTracker() as c1:
        ds = get_lance_dataset(settings.lance_volume_path, settings)
        table = ds.to_table(columns=VOLUME_FEATURE_COLS + [VOLUME_TARGET_COL])

    results["data_load"] = BenchmarkResult(
        wall_clock_sec=t1.elapsed_sec,
        peak_memory_mb=m1.peak_mb,
        cpu_percent=c1.cpu_percent,
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
        wall_clock_sec=t2.elapsed_sec,
        peak_memory_mb=m2.peak_mb,
        cpu_percent=c2.cpu_percent,
    )

    # Step 3: Train 5 epochs
    with BenchmarkTimer() as t3, MemoryTracker() as m3, CpuTracker() as c3:
        train_mlp(dataloader, input_dim=len(VOLUME_FEATURE_COLS), epochs=5)

    results["train_5_epochs"] = BenchmarkResult(
        wall_clock_sec=t3.elapsed_sec,
        peak_memory_mb=m3.peak_mb,
        cpu_percent=c3.cpu_percent,
    )

    # Total
    total_time = t1.elapsed_sec + t2.elapsed_sec + t3.elapsed_sec
    total_mem = max(m1.peak_mb, m2.peak_mb, m3.peak_mb)
    results["total"] = BenchmarkResult(
        wall_clock_sec=total_time,
        peak_memory_mb=total_mem,
    )

    return results


def bench_pytorch_parquet(settings: Settings) -> dict[str, BenchmarkResult]:
    """Benchmark PyTorch MLP training with Parquet data source."""
    results = {}

    # Step 1: Data load
    with BenchmarkTimer() as t1, MemoryTracker() as m1, CpuTracker() as c1:
        ds = get_parquet_dataset(settings.parquet_volume_path, settings)
        table = ds.to_table(columns=VOLUME_FEATURE_COLS + [VOLUME_TARGET_COL])

    results["data_load"] = BenchmarkResult(
        wall_clock_sec=t1.elapsed_sec,
        peak_memory_mb=m1.peak_mb,
        cpu_percent=c1.cpu_percent,
        rows_read=table.num_rows,
    )

    # Step 2: DataLoader init
    with BenchmarkTimer() as t2, MemoryTracker() as m2, CpuTracker() as c2:
        X = table.select(VOLUME_FEATURE_COLS).to_pandas().to_numpy(dtype=np.float32)
        y = table.column(VOLUME_TARGET_COL).to_numpy().astype(np.float32)
        dataloader = create_dataloader_from_numpy(X, y, batch_size=1024, shuffle=True)

    results["dataloader_init"] = BenchmarkResult(
        wall_clock_sec=t2.elapsed_sec,
        peak_memory_mb=m2.peak_mb,
        cpu_percent=c2.cpu_percent,
    )

    # Step 3: Train 5 epochs
    with BenchmarkTimer() as t3, MemoryTracker() as m3, CpuTracker() as c3:
        train_mlp(dataloader, input_dim=len(VOLUME_FEATURE_COLS), epochs=5)

    results["train_5_epochs"] = BenchmarkResult(
        wall_clock_sec=t3.elapsed_sec,
        peak_memory_mb=m3.peak_mb,
        cpu_percent=c3.cpu_percent,
    )

    # Total
    total_time = t1.elapsed_sec + t2.elapsed_sec + t3.elapsed_sec
    total_mem = max(m1.peak_mb, m2.peak_mb, m3.peak_mb)
    results["total"] = BenchmarkResult(
        wall_clock_sec=total_time,
        peak_memory_mb=total_mem,
    )

    return results
