"""Project configuration and constants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """All benchmark settings in one place."""

    # MinIO connection
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    bucket_name: str = "benchmark"

    # S3 paths
    lance_decision_path: str = "s3://benchmark/lance/decision"
    parquet_decision_path: str = "s3://benchmark/parquet/decision"
    lance_volume_path: str = "s3://benchmark/lance/volume"
    parquet_volume_path: str = "s3://benchmark/parquet/volume"

    # Data scale
    num_sites: int = 100
    num_tanks_per_site: int = 3
    num_days: int = 365
    seed: int = 42

    # Benchmark params
    benchmark_repeats: int = 3
    warmup_runs: int = 1

    # Column subsets for benchmark 3
    column_subset: tuple[str, ...] = (
        "overdue_ratio",
        "inv_days_cover",
        "dow_hist_rate",
        "hist_rate",
        "avg_sale_7d",
        "open_inventory",
        "tank_capacity",
        "delivery_occurred",
    )

    # Random access params
    random_access_sample_size: int = 10_000

    @property
    def storage_options(self) -> dict[str, str]:
        """S3 storage options for lance and pyarrow."""
        return {
            "aws_access_key_id": self.minio_access_key,
            "aws_secret_access_key": self.minio_secret_key,
            "endpoint_url": self.minio_endpoint,
            "allow_http": "true",
            "region": "us-east-1",
        }

    @property
    def total_decision_rows(self) -> int:
        return self.num_sites * self.num_tanks_per_site * self.num_days

    @property
    def total_volume_rows(self) -> int:
        return self.num_sites * self.num_tanks_per_site * self.num_days * 24
