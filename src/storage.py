"""Storage utilities: write/read Lance and Parquet to/from MinIO."""

from __future__ import annotations

import lance
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import s3fs

from src.config import Settings


def _get_s3fs(settings: Settings) -> s3fs.S3FileSystem:
    """Create an S3FileSystem configured for MinIO."""
    return s3fs.S3FileSystem(
        key=settings.minio_access_key,
        secret=settings.minio_secret_key,
        endpoint_url=settings.minio_endpoint,
        use_ssl=False,
    )


def ensure_bucket(settings: Settings) -> None:
    """Create the benchmark bucket if it doesn't exist."""
    fs = _get_s3fs(settings)
    if not fs.exists(settings.bucket_name):
        fs.mkdir(settings.bucket_name)


def _clean_s3_path(fs: s3fs.S3FileSystem, s3_path: str, settings: Settings) -> None:
    """Remove all objects under an S3 path prefix using boto3."""
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
    )
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=settings.bucket_name, Prefix=s3_path):
            for obj in page.get("Contents", []):
                s3.delete_object(Bucket=settings.bucket_name, Key=obj["Key"])
    except Exception:
        pass


def write_parquet_to_minio(
    table: pa.Table,
    path: str,
    settings: Settings,
    partition_col: str | None = "feature_date",
) -> None:
    """Write a PyArrow Table as a partitioned Parquet dataset to MinIO."""
    fs = _get_s3fs(settings)
    # Strip s3:// prefix for s3fs
    s3_path = path.replace("s3://", "")

    # Clean existing data to prevent accumulation
    _clean_s3_path(fs, s3_path, settings)

    if partition_col and partition_col in table.column_names:
        pq.write_to_dataset(
            table,
            root_path=s3_path,
            partition_cols=[partition_col],
            filesystem=fs,
        )
    else:
        pq.write_table(table, f"{s3_path}/data.parquet", filesystem=fs)


def write_lance_to_minio(table: pa.Table, path: str, settings: Settings) -> None:
    """Write a PyArrow Table as a Lance dataset to MinIO."""
    storage_options = {
        "aws_access_key_id": settings.minio_access_key,
        "aws_secret_access_key": settings.minio_secret_key,
        "aws_endpoint": settings.minio_endpoint,
        "allow_http": "true",
        "aws_region": "us-east-1",
    }
    lance.write_dataset(table, path, storage_options=storage_options, mode="overwrite")


def get_lance_dataset(path: str, settings: Settings) -> lance.LanceDataset:
    """Open a Lance dataset from MinIO."""
    storage_options = {
        "aws_access_key_id": settings.minio_access_key,
        "aws_secret_access_key": settings.minio_secret_key,
        "aws_endpoint": settings.minio_endpoint,
        "allow_http": "true",
        "aws_region": "us-east-1",
    }
    return lance.dataset(path, storage_options=storage_options)


def get_parquet_dataset(path: str, settings: Settings) -> pds.Dataset:
    """Open a Parquet dataset from MinIO."""
    fs = _get_s3fs(settings)
    s3_path = path.replace("s3://", "")
    return pds.dataset(s3_path, filesystem=fs, format="parquet")


def get_dataset_size_mb(path: str, settings: Settings) -> float:
    """Get total size of all files under a path in MB."""
    fs = _get_s3fs(settings)
    s3_path = path.replace("s3://", "")
    total_bytes = 0
    try:
        for root, _dirs, files in fs.walk(s3_path):
            for f in files:
                full_path = f"{root}/{f}" if not root.endswith("/") else f"{root}{f}"
                info = fs.info(full_path)
                total_bytes += info.get("size", 0)
    except FileNotFoundError:
        return 0.0
    return total_bytes / (1024 * 1024)
