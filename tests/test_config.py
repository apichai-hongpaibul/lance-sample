from src.config import Settings


def test_settings_defaults():
    s = Settings()
    assert s.minio_endpoint == "http://localhost:9000"
    assert s.minio_access_key == "minioadmin"
    assert s.minio_secret_key == "minioadmin"
    assert s.bucket_name == "benchmark"
    assert s.num_sites == 500
    assert s.num_tanks_per_site == 3
    assert s.num_days == 365
    assert s.seed == 42
    assert s.benchmark_repeats == 3
    assert s.warmup_runs == 1


def test_settings_paths():
    s = Settings()
    assert s.lance_decision_path == "s3://benchmark/lance/decision"
    assert s.parquet_decision_path == "s3://benchmark/parquet/decision"
    assert s.lance_volume_path == "s3://benchmark/lance/volume"
    assert s.parquet_volume_path == "s3://benchmark/parquet/volume"


def test_settings_storage_options():
    s = Settings()
    opts = s.storage_options
    assert opts["aws_access_key_id"] == "minioadmin"
    assert opts["aws_secret_access_key"] == "minioadmin"
    assert opts["endpoint_url"] == "http://localhost:9000"
    assert opts["allow_http"] == "true"


def test_settings_total_rows():
    s = Settings()
    assert s.total_decision_rows == 500 * 3 * 365
    assert s.total_volume_rows == 500 * 3 * 365 * 24
