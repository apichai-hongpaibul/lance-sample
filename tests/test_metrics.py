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
        data = [0] * 100_000  # noqa: F841
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
