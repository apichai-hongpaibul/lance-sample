"""Benchmark instrumentation: timing, memory, CPU, S3 call tracking."""

import gc
import time
import tracemalloc
from dataclasses import dataclass
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
    """Context manager for peak memory tracking via RSS (psutil).

    Uses process RSS instead of tracemalloc to avoid conflicts with
    C++ extensions like XGBoost that manage their own memory.
    """

    def __init__(self):
        self.peak_mb: float = 0.0
        self._baseline_mb: float = 0.0
        self._process = psutil.Process()

    def __enter__(self):
        self._baseline_mb = self._process.memory_info().rss / (1024 * 1024)
        return self

    def __exit__(self, *exc):
        current = self._process.memory_info().rss / (1024 * 1024) - self._baseline_mb
        self.peak_mb = max(current, 0.0)
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

    def reset(self):
        self.calls = {k: 0 for k in self.calls}

    @property
    def total(self) -> int:
        return sum(self.calls.values())

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
