import pyarrow as pa

from src.config import Settings
from src.generate import generate_decision_features, generate_volume_events


def test_decision_features_shape():
    s = Settings(num_sites=5, num_tanks_per_site=2, num_days=3)
    table = generate_decision_features(s)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 5 * 2 * 3  # 30 rows
    assert table.num_columns >= 30


def test_decision_features_columns():
    s = Settings(num_sites=2, num_tanks_per_site=1, num_days=2)
    table = generate_decision_features(s)
    col_names = table.column_names
    assert "site_code" in col_names
    assert "tank_id" in col_names
    assert "feature_date" in col_names
    assert "overdue_ratio" in col_names
    assert "delivery_occurred" in col_names
    assert "inv_days_cover" in col_names
    assert "dow_hist_rate" in col_names


def test_decision_features_reproducible():
    s = Settings(num_sites=3, num_tanks_per_site=1, num_days=2)
    t1 = generate_decision_features(s)
    t2 = generate_decision_features(s)
    assert t1.equals(t2)


def test_decision_features_target_distribution():
    s = Settings(num_sites=10, num_tanks_per_site=2, num_days=30)
    table = generate_decision_features(s)
    target = table.column("delivery_occurred").to_pylist()
    # Should have both 0s and 1s
    assert 0 in target
    assert 1 in target


def test_volume_events_shape():
    s = Settings(num_sites=3, num_tanks_per_site=2, num_days=2)
    table = generate_volume_events(s)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 3 * 2 * 2 * 24  # 288 rows
    assert table.num_columns == 6


def test_volume_events_columns():
    s = Settings(num_sites=2, num_tanks_per_site=1, num_days=1)
    table = generate_volume_events(s)
    expected_cols = {
        "reading_date",
        "reading_hour",
        "site_code",
        "tank_id",
        "atg_start",
        "atg_diff",
    }
    assert set(table.column_names) == expected_cols


def test_volume_events_hour_range():
    s = Settings(num_sites=1, num_tanks_per_site=1, num_days=1)
    table = generate_volume_events(s)
    hours = table.column("reading_hour").to_pylist()
    assert min(hours) == 0
    assert max(hours) == 23


def test_volume_events_reproducible():
    s = Settings(num_sites=2, num_tanks_per_site=1, num_days=2)
    t1 = generate_volume_events(s)
    t2 = generate_volume_events(s)
    assert t1.equals(t2)
