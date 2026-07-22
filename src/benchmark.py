"""Main benchmark orchestrator — generates data, runs benchmarks, produces report."""

import gc
import platform
import statistics
import sys
from typing import Callable

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
    steps = results_list[0].keys()
    summary = {}
    for step in steps:
        times = [r[step].wall_clock_sec for r in results_list]
        summary[step] = statistics.mean(times)
    return summary


def _run_training_benchmark(
    fn: Callable[[], dict[str, BenchmarkResult]],
    repeats: int = 3,
    warmup: int = 1,
) -> list[dict[str, BenchmarkResult]]:
    """Run a training benchmark function that returns dict[str, BenchmarkResult].

    Similar to run_benchmark but for functions returning step-level dicts.
    """
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
        decision_table,
        settings.parquet_decision_path,
        settings,
        partition_col="feature_date",
    )
    write_parquet_to_minio(
        volume_table, settings.parquet_volume_path, settings, partition_col=None
    )

    print("  Writing Lance to MinIO...")
    write_lance_to_minio(decision_table, settings.lance_decision_path, settings)
    write_lance_to_minio(volume_table, settings.lance_volume_path, settings)

    # Get file sizes
    file_sizes = {
        "lance_decision_mb": get_dataset_size_mb(
            settings.lance_decision_path, settings
        ),
        "parquet_decision_mb": get_dataset_size_mb(
            settings.parquet_decision_path, settings
        ),
        "lance_volume_mb": get_dataset_size_mb(settings.lance_volume_path, settings),
        "parquet_volume_mb": get_dataset_size_mb(
            settings.parquet_volume_path, settings
        ),
    }
    print(
        f"  File sizes: Lance Decision={file_sizes['lance_decision_mb']:.1f}MB, "
        f"Parquet Decision={file_sizes['parquet_decision_mb']:.1f}MB"
    )

    # Step 3: Access pattern benchmarks
    print(
        f"\n[3/6] Running access pattern benchmarks "
        f"({settings.warmup_runs} warmup + {settings.benchmark_repeats} measured)..."
    )

    print("  Random access (Lance)...")
    lance_ra = run_benchmark(
        lambda: bench_random_access_lance(settings),
        repeats=settings.benchmark_repeats,
        warmup=settings.warmup_runs,
    )
    print("  Random access (Parquet)...")
    parquet_ra = run_benchmark(
        lambda: bench_random_access_parquet(settings),
        repeats=settings.benchmark_repeats,
        warmup=settings.warmup_runs,
    )

    print("  Sequential scan (Lance)...")
    lance_seq = run_benchmark(
        lambda: bench_sequential_scan_lance(settings),
        repeats=settings.benchmark_repeats,
        warmup=settings.warmup_runs,
    )
    print("  Sequential scan (Parquet)...")
    parquet_seq = run_benchmark(
        lambda: bench_sequential_scan_parquet(settings),
        repeats=settings.benchmark_repeats,
        warmup=settings.warmup_runs,
    )

    print("  Column subset (Lance)...")
    lance_col = run_benchmark(
        lambda: bench_column_subset_lance(settings),
        repeats=settings.benchmark_repeats,
        warmup=settings.warmup_runs,
    )
    print("  Column subset (Parquet)...")
    parquet_col = run_benchmark(
        lambda: bench_column_subset_parquet(settings),
        repeats=settings.benchmark_repeats,
        warmup=settings.warmup_runs,
    )

    # Step 4: Training benchmarks
    print("\n[4/6] Running XGBoost training benchmark...")
    lance_xgb = _run_training_benchmark(
        lambda: bench_xgboost_lance(settings),
        repeats=settings.benchmark_repeats,
        warmup=settings.warmup_runs,
    )
    parquet_xgb = _run_training_benchmark(
        lambda: bench_xgboost_parquet(settings),
        repeats=settings.benchmark_repeats,
        warmup=settings.warmup_runs,
    )

    print("\n[5/6] Running PyTorch training benchmark...")
    lance_pt = _run_training_benchmark(
        lambda: bench_pytorch_lance(settings),
        repeats=settings.benchmark_repeats,
        warmup=settings.warmup_runs,
    )
    parquet_pt = _run_training_benchmark(
        lambda: bench_pytorch_parquet(settings),
        repeats=settings.benchmark_repeats,
        warmup=settings.warmup_runs,
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
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Random access: Lance {ra_speedup:.1f}x faster")
    print("  Report saved to: report.html")
    print("  Raw data saved to: results.json")
    print(f"{'=' * 60}")
