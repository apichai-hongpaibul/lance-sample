"""Storage tests — require MinIO running (docker compose up -d)."""

import datetime

import pyarrow as pa
import pytest

from src.config import Settings
from src.storage import (
    ensure_bucket,
    get_dataset_size_mb,
    get_lance_dataset,
    get_parquet_dataset,
    write_lance_to_minio,
    write_parquet_to_minio,
)

# Use test-specific paths to avoid collision
TEST_SETTINGS = Settings(
    lance_decision_path="s3://benchmark/test/lance/decision",
    parquet_decision_path="s3://benchmark/test/parquet/decision",
)


@pytest.fixture(scope="module")
def sample_table():
    """Small PyArrow table for testing."""
    return pa.table(
        {
            "site_code": ["I001", "I001", "I002", "I002"],
            "tank_id": ["T01", "T01", "T01", "T01"],
            "feature_date": [
                datetime.date(2025, 7, 22),
                datetime.date(2025, 7, 23),
                datetime.date(2025, 7, 22),
                datetime.date(2025, 7, 23),
            ],
            "overdue_ratio": [1.5, 0.8, 2.1, 0.3],
            "delivery_occurred": pa.array([1, 0, 1, 0], type=pa.int8()),
        }
    )


@pytest.fixture(scope="module", autouse=True)
def setup_bucket():
    """Ensure bucket exists before tests."""
    ensure_bucket(TEST_SETTINGS)


@pytest.mark.integration
def test_write_and_read_parquet(sample_table):
    path = TEST_SETTINGS.parquet_decision_path
    write_parquet_to_minio(sample_table, path, TEST_SETTINGS, partition_col="feature_date")

    ds = get_parquet_dataset(path, TEST_SETTINGS)
    result = ds.to_table()
    assert result.num_rows == 4


@pytest.mark.integration
def test_write_and_read_lance(sample_table):
    path = TEST_SETTINGS.lance_decision_path
    write_lance_to_minio(sample_table, path, TEST_SETTINGS)

    ds = get_lance_dataset(path, TEST_SETTINGS)
    result = ds.to_table()
    assert result.num_rows == 4


@pytest.mark.integration
def test_get_dataset_size(sample_table):
    path = TEST_SETTINGS.parquet_decision_path
    write_parquet_to_minio(sample_table, path, TEST_SETTINGS, partition_col="feature_date")
    size = get_dataset_size_mb(path, TEST_SETTINGS)
    assert size > 0
