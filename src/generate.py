"""Synthetic data generation for GAIA-style feature tables."""

import datetime

import numpy as np
import pyarrow as pa

from src.config import Settings

# Products and regions from the GAIA spec
PRODUCTS = ["5000018", "5000011", "5000012"]  # HSD, ULG91, ULG95
REGIONS = ["NORTH", "SOUTH", "EAST", "WEST", "CENTRAL"]


def generate_decision_features(settings: Settings) -> pa.Table:
    """Generate the Decision Model feature table (wide, 30+ columns).

    Produces realistic correlated features matching the GAIA Gold
    gold_decision_model_features schema.
    """
    rng = np.random.default_rng(settings.seed)

    n_rows = settings.total_decision_rows
    n_sites = settings.num_sites
    n_tanks = settings.num_tanks_per_site
    n_days = settings.num_days

    # Generate keys by tiling
    site_codes = [f"I{i:03d}" for i in range(1, n_sites + 1)]
    tank_ids = [f"T{t:02d}" for t in range(1, n_tanks + 1)]

    # Create all combinations: site × tank × day
    start_date = datetime.date(2025, 7, 22)
    dates = [start_date + datetime.timedelta(days=d) for d in range(n_days)]

    # Build arrays by repeating patterns
    site_arr = np.repeat(site_codes, n_tanks * n_days)
    tank_arr = np.tile(np.repeat(tank_ids, n_days), n_sites)
    date_arr = np.tile(dates, n_sites * n_tanks)

    # Assign fixed attributes per site
    site_regions = rng.choice(REGIONS, size=n_sites)
    region_arr = np.repeat(site_regions, n_tanks * n_days)

    site_products = rng.choice(PRODUCTS, size=n_sites)
    product_arr = np.repeat(site_products, n_tanks * n_days)

    # Tank capacity (fixed per site-tank)
    tank_capacities = rng.uniform(15000, 40000, size=n_sites * n_tanks)
    tank_capacity_arr = np.repeat(tank_capacities, n_days)

    # Generate correlated features
    # Base delivery probability influenced by overdue_ratio
    base_overdue = rng.exponential(0.8, size=n_rows)
    overdue_ratio = np.clip(base_overdue, 0.0, 3.0)

    # Higher overdue → more likely to deliver
    delivery_prob = 1 / (1 + np.exp(-(overdue_ratio - 1.0)))
    delivery_occurred = rng.binomial(1, delivery_prob).astype(np.int8)

    # Inventory features (anti-correlated with overdue)
    inv_days_cover = np.clip(
        rng.normal(7.0, 3.0, size=n_rows) - overdue_ratio * 2, 0.0, 15.0
    )
    open_inventory = tank_capacity_arr * rng.uniform(0.1, 0.9, size=n_rows)

    # Sales features
    avg_sale_7d = rng.uniform(500, 5000, size=n_rows)
    avg_sale_30d = avg_sale_7d * rng.uniform(0.8, 1.2, size=n_rows)

    # Historical rates with day-of-week seasonality
    day_of_week = np.array([d.weekday() for d in date_arr], dtype=np.int8)
    # Weekend has lower delivery rate
    weekend_factor = np.where((day_of_week == 5) | (day_of_week == 6), 0.3, 0.7)
    dow_hist_rate = np.clip(
        weekend_factor + rng.normal(0, 0.1, size=n_rows), 0.0, 1.0
    )
    hist_rate = np.clip(rng.beta(2, 3, size=n_rows), 0.0, 1.0)

    # Delivery volume features
    delivery_sum_28d = rng.uniform(0, 200000, size=n_rows)
    proj_end_fill_ratio = np.clip(rng.beta(3, 2, size=n_rows), 0.0, 1.0)

    # Percentile features
    group_overdue_pct = rng.uniform(0.0, 1.0, size=n_rows)
    group_cover_pct = rng.uniform(0.0, 1.0, size=n_rows)

    # Other features
    last_delivery_days_ago = rng.integers(0, 31, size=n_rows).astype(np.int32)
    intransit_volume = rng.uniform(0, 30000, size=n_rows)
    usage_day_at_approval = rng.uniform(0.5, 10.0, size=n_rows)
    current_inventory_at_approval = open_inventory * rng.uniform(
        0.8, 1.0, size=n_rows
    )

    # Date features
    day_of_month = np.array([d.day for d in date_arr], dtype=np.int8)
    month = np.array([d.month for d in date_arr], dtype=np.int8)
    is_weekend = (day_of_week == 5) | (day_of_week == 6)
    is_holiday = rng.random(size=n_rows) < 0.03  # ~3% holidays

    # Lag features
    delivery_volume_lag1 = rng.uniform(0, 30000, size=n_rows)
    delivery_volume_lag7 = rng.uniform(0, 30000, size=n_rows)
    rolling_delivery_count_7d = rng.integers(0, 8, size=n_rows).astype(np.int32)
    rolling_delivery_count_30d = rng.integers(0, 31, size=n_rows).astype(np.int32)

    # Pair features
    pair_deliv_mean = rng.uniform(5000, 25000, size=n_rows)
    yoy_deliv = rng.uniform(0, 50000, size=n_rows)

    # Build PyArrow Table
    table = pa.table(
        {
            "site_code": pa.array(site_arr, type=pa.string()),
            "tank_id": pa.array(tank_arr, type=pa.string()),
            "feature_date": pa.array(date_arr, type=pa.date32()),
            "product_code": pa.array(product_arr, type=pa.string()),
            "region_code": pa.array(region_arr, type=pa.string()),
            "tank_capacity": pa.array(tank_capacity_arr, type=pa.float64()),
            "open_inventory": pa.array(open_inventory, type=pa.float64()),
            "avg_sale_7d": pa.array(avg_sale_7d, type=pa.float64()),
            "avg_sale_30d": pa.array(avg_sale_30d, type=pa.float64()),
            "overdue_ratio": pa.array(overdue_ratio, type=pa.float64()),
            "inv_days_cover": pa.array(inv_days_cover, type=pa.float64()),
            "dow_hist_rate": pa.array(dow_hist_rate, type=pa.float64()),
            "hist_rate": pa.array(hist_rate, type=pa.float64()),
            "delivery_sum_28d": pa.array(delivery_sum_28d, type=pa.float64()),
            "proj_end_fill_ratio": pa.array(
                proj_end_fill_ratio, type=pa.float64()
            ),
            "group_overdue_pct": pa.array(group_overdue_pct, type=pa.float64()),
            "group_cover_pct": pa.array(group_cover_pct, type=pa.float64()),
            "last_delivery_days_ago": pa.array(
                last_delivery_days_ago, type=pa.int32()
            ),
            "intransit_volume": pa.array(intransit_volume, type=pa.float64()),
            "usage_day_at_approval": pa.array(
                usage_day_at_approval, type=pa.float64()
            ),
            "current_inventory_at_approval": pa.array(
                current_inventory_at_approval, type=pa.float64()
            ),
            "day_of_week": pa.array(day_of_week, type=pa.int8()),
            "day_of_month": pa.array(day_of_month, type=pa.int8()),
            "month": pa.array(month, type=pa.int8()),
            "is_weekend": pa.array(is_weekend, type=pa.bool_()),
            "is_holiday": pa.array(is_holiday, type=pa.bool_()),
            "delivery_volume_lag1": pa.array(
                delivery_volume_lag1, type=pa.float64()
            ),
            "delivery_volume_lag7": pa.array(
                delivery_volume_lag7, type=pa.float64()
            ),
            "rolling_delivery_count_7d": pa.array(
                rolling_delivery_count_7d, type=pa.int32()
            ),
            "rolling_delivery_count_30d": pa.array(
                rolling_delivery_count_30d, type=pa.int32()
            ),
            "pair_deliv_mean": pa.array(pair_deliv_mean, type=pa.float64()),
            "yoy_deliv": pa.array(yoy_deliv, type=pa.float64()),
            "delivery_occurred": pa.array(delivery_occurred, type=pa.int8()),
        }
    )
    return table


def generate_volume_events(settings: Settings) -> pa.Table:
    """Generate the Volume Model hourly events table (narrow, time-series).

    Produces ATG hourly readings with realistic daily consumption patterns.
    """
    rng = np.random.default_rng(settings.seed + 1)  # different seed from decision

    n_sites = settings.num_sites
    n_tanks = settings.num_tanks_per_site
    n_days = settings.num_days
    n_rows = settings.total_volume_rows

    # Keys
    site_codes = [f"I{i:03d}" for i in range(1, n_sites + 1)]
    tank_ids = [f"T{t:02d}" for t in range(1, n_tanks + 1)]
    start_date = datetime.date(2025, 7, 22)
    dates = [start_date + datetime.timedelta(days=d) for d in range(n_days)]
    hours = list(range(24))

    # Build arrays: site × tank × day × hour
    site_arr = np.repeat(site_codes, n_tanks * n_days * 24)
    tank_arr = np.tile(np.repeat(tank_ids, n_days * 24), n_sites)
    date_arr = np.tile(np.repeat(dates, 24), n_sites * n_tanks)
    hour_arr = np.tile(hours, n_sites * n_tanks * n_days).astype(np.int8)

    # ATG consumption pattern: peak at 7-9 and 17-19
    hour_weights = np.array(
        [
            0.2, 0.1, 0.1, 0.1, 0.2, 0.3,  # 0-5: night, low
            0.5, 0.9, 1.0, 0.8, 0.7, 0.6,  # 6-11: morning peak
            0.5, 0.5, 0.5, 0.6, 0.7, 0.9,  # 12-17: afternoon rise
            1.0, 0.8, 0.6, 0.4, 0.3, 0.2,  # 18-23: evening peak then decline
        ]
    )
    base_consumption = hour_weights[hour_arr] * rng.uniform(50, 300, size=n_rows)
    atg_diff = np.abs(base_consumption + rng.normal(0, 20, size=n_rows))

    # ATG start: tank level that decreases through the day
    tank_capacity_per_combo = rng.uniform(15000, 40000, size=n_sites * n_tanks)
    daily_start = np.repeat(tank_capacity_per_combo, n_days * 24) * rng.uniform(
        0.3, 0.9, size=n_rows
    )
    atg_start = np.clip(daily_start - atg_diff * (hour_arr / 24.0), 1000, 40000)

    table = pa.table(
        {
            "reading_date": pa.array(date_arr, type=pa.date32()),
            "reading_hour": pa.array(hour_arr, type=pa.int8()),
            "site_code": pa.array(site_arr, type=pa.string()),
            "tank_id": pa.array(tank_arr, type=pa.string()),
            "atg_start": pa.array(atg_start, type=pa.float64()),
            "atg_diff": pa.array(atg_diff, type=pa.float64()),
        }
    )
    return table
