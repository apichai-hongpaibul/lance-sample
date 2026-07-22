"""Report generation tests."""

import json
import os
import tempfile

from src.report import generate_report, save_results_json


def _mock_results():
    """Create mock benchmark results for testing."""
    return {
        "environment": {
            "python_version": "3.11.0",
            "cpu": "Apple M1",
            "ram_gb": 16,
            "os": "macOS",
        },
        "file_sizes": {
            "lance_decision_mb": 45.2,
            "parquet_decision_mb": 38.1,
            "lance_volume_mb": 120.5,
            "parquet_volume_mb": 105.3,
        },
        "access_patterns": {
            "random_access": {
                "lance": {
                    "wall_clock_mean": 0.15,
                    "wall_clock_std": 0.02,
                    "peak_memory_mb": 50.0,
                    "cpu_percent": 45.0,
                },
                "parquet": {
                    "wall_clock_mean": 2.5,
                    "wall_clock_std": 0.3,
                    "peak_memory_mb": 200.0,
                    "cpu_percent": 60.0,
                },
            },
            "sequential_scan": {
                "lance": {
                    "wall_clock_mean": 1.2,
                    "wall_clock_std": 0.1,
                    "peak_memory_mb": 300.0,
                    "cpu_percent": 70.0,
                },
                "parquet": {
                    "wall_clock_mean": 1.1,
                    "wall_clock_std": 0.1,
                    "peak_memory_mb": 280.0,
                    "cpu_percent": 65.0,
                },
            },
            "column_subset": {
                "lance": {
                    "wall_clock_mean": 0.5,
                    "wall_clock_std": 0.05,
                    "peak_memory_mb": 80.0,
                    "cpu_percent": 40.0,
                },
                "parquet": {
                    "wall_clock_mean": 0.6,
                    "wall_clock_std": 0.06,
                    "peak_memory_mb": 90.0,
                    "cpu_percent": 42.0,
                },
            },
        },
        "training": {
            "xgboost": {
                "lance": {
                    "data_load": 0.8,
                    "to_dmatrix": 0.3,
                    "train": 5.0,
                    "total": 6.1,
                },
                "parquet": {
                    "data_load": 1.2,
                    "to_dmatrix": 0.3,
                    "train": 5.0,
                    "total": 6.5,
                },
            },
            "pytorch": {
                "lance": {
                    "data_load": 0.5,
                    "dataloader_init": 0.2,
                    "train_5_epochs": 12.0,
                    "total": 12.7,
                },
                "parquet": {
                    "data_load": 4.5,
                    "dataloader_init": 1.5,
                    "train_5_epochs": 12.0,
                    "total": 18.0,
                },
            },
        },
    }


def test_generate_report_creates_file():
    results = _mock_results()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        generate_report(results, output_path=path)
        assert os.path.exists(path)
        content = open(path).read()
        assert "<html" in content
        assert "Lance" in content
        assert "Parquet" in content
        assert "plotly" in content.lower() or "Plotly" in content
    finally:
        os.unlink(path)


def test_generate_report_contains_sections():
    results = _mock_results()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        generate_report(results, output_path=path)
        content = open(path).read()
        assert "Summary" in content
        assert "Random" in content or "random" in content
        assert "Sequential" in content or "sequential" in content
        assert "XGBoost" in content or "xgboost" in content
        assert "PyTorch" in content or "pytorch" in content
    finally:
        os.unlink(path)


def test_save_results_json():
    results = _mock_results()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        save_results_json(results, output_path=path)
        assert os.path.exists(path)
        loaded = json.loads(open(path).read())
        assert "environment" in loaded
        assert "access_patterns" in loaded
        assert "training" in loaded
    finally:
        os.unlink(path)
