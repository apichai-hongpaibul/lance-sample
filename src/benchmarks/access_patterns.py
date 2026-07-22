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
    indices = rng.choice(
        n_rows, size=min(settings.random_access_sample_size, n_rows), replace=False
    )
    indices = sorted(indices.tolist())

    with BenchmarkTimer() as timer, MemoryTracker() as mem, CpuTracker() as cpu:
        # Lance supports take() for O(1) random access
        table = ds.take(indices)

    return BenchmarkResult(
        wall_clock_sec=timer.elapsed_sec,
        peak_memory_mb=mem.peak_mb,
        cpu_percent=cpu.cpu_percent,
        rows_read=table.num_rows,
        first_row_latency_sec=timer.elapsed_sec,  # approximate for take()
    )


def bench_random_access_parquet(settings: Settings) -> BenchmarkResult:
    """Benchmark: read random rows from Parquet dataset."""
    ds = get_parquet_dataset(settings.parquet_decision_path, settings)
    # Need to know total rows for random index generation
    # Read metadata/count first (outside timing)
    scanner = ds.scanner()
    n_rows = scanner.count_rows()
    rng = np.random.default_rng(settings.seed)
    indices = rng.choice(
        n_rows, size=min(settings.random_access_sample_size, n_rows), replace=False
    )
    indices = sorted(indices.tolist())

    with BenchmarkTimer() as timer, MemoryTracker() as mem, CpuTracker() as cpu:
        # Parquet: must read full table then take — no native random access
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
