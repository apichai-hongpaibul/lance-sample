# PRD — GAIA → AI Data Platform (Iceberg / Airflow / dbt / Trino)

| | |
|---|---|
| Status | Draft for review |
| Owners | Data Engineering (pipeline), AI/ML Engineering (features), GAIA app team (source API) |
| Related docs | `GAIA_Data_Platform_Integration_Spec.md` (architecture, REST API contract, ER model) |

---

## 1. Summary

Build the ingestion, storage, and transform layer that moves GAIA's transactional and master data into an Iceberg-on-MinIO lakehouse, on a midnight EOD schedule for transactions and a change-triggered schedule for master data, so AI training (Decision Model, Volume Model) and BI (Superset, Power BI) read from one governed source instead of ad-hoc exports.

## 2. Goals / non-goals

**Goals**
- Deterministic, replayable Airflow extraction from GAIA's REST API into Bronze.
- dbt-owned, version-controlled Bronze → Silver → Gold transforms, including the anomaly flags and feature formulas already defined by the AI team.
- Query performance on tank + date filters, since nearly every downstream feature (`pair_deliv_mean`, `hist_rate`, `dow_hist_rate`) groups by `(site_code, tank_id, date)`.
- One serving layer (Trino) for BI and ML, so dashboard numbers and model features never diverge.

**Non-goals (this phase)**
- Real-time inference plumbing (Item 29's "Call Service" real-time path) — tracked separately.
- Truck Load Optimizer service itself — this PRD covers data plumbing only.

## 3. Storage layout and partitioning

**Ask addressed: group by tank/date for read performance.**

Two options were considered:

| Option | Partition spec | Tradeoff |
|---|---|---|
| A — identity partition on `tank_id` | `partitioned by (tank_id, days(event_date))` | Simple, but with thousands of tanks this creates a very large number of small partitions — Iceberg/Trino planning overhead grows, small-file problem gets worse. |
| **B — bucketed tank + daily date (recommended)** | `partitioned by (days(event_date), bucket(tank_id, 16))` + files **sorted by** `(site_code, tank_id, event_date)` | Keeps partition count bounded (≈365 × 16/year), while sort order gives Parquet row-group min/max stats that let Trino/dbt skip row groups for a specific tank within a date partition — the actual thing that makes `WHERE tank_id = 'T01' AND event_date BETWEEN ...` fast. |

Use **Option B** for all six transactional tables (Items 1–6). Master data (Item 0) is small — no partitioning needed, just SCD2 columns.

```sql
-- Bronze table DDL (via pyiceberg or Trino), e.g. possale_hourly
CREATE TABLE bronze.gaia.possale_hourly (
  sale_date        date,
  sale_hour        integer,
  site_code        varchar,
  tank_id          varchar,
  product_code     varchar,
  total_hourly_sale_volume decimal(12,2),
  _ingested_at      timestamp(6),
  _source_batch_id  varchar
)
WITH (
  format = 'PARQUET',
  partitioning = ARRAY['day(sale_date)', 'bucket(tank_id, 16)'],
  sorted_by = ARRAY['site_code', 'tank_id', 'sale_date']
);
```

Bucket count of 16 is a starting point — tune against actual tank cardinality once known (see open questions §9).

---

## 4. Python extraction scripts (Airflow)

### 4.1 Project layout

```
gaia_pipeline/
  dags/
    gaia_transaction_eod_extract.py
    gaia_master_data_sync.py
  gaia_pipeline/
    __init__.py
    config.py            # env-driven settings (Airflow Variables / .env)
    gaia_client.py        # REST client: auth, retry, pagination
    extractors/
      base.py             # shared extract-and-land logic, watermarking
      order_product.py     # Item 1
      order_site_tank.py   # Item 2
      truck_allocation.py  # Item 3
      possale_daily.py     # Item 4
      possale_hourly.py    # Item 5
      atg_hourly.py         # Item 6
      master_sites.py       # Item 0 (change-triggered)
    writer.py              # Arrow table -> Iceberg write (pyiceberg)
    watermark.py            # get/set last-extracted watermark
  tests/
  pyproject.toml            # uv-managed deps
```

Dependency management: `uv`, consistent with the existing DataMart pipeline conventions. Core deps: `pyiceberg`, `pyarrow`, `httpx` (retry + async-capable), `pydantic` (response validation), `apache-airflow-providers-common-sql`.

### 4.2 GAIA REST client

```python
# gaia_pipeline/gaia_client.py
import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class GaiaClientConfig(BaseModel):
    base_url: str
    api_key: str
    timeout_seconds: float = 30.0
    max_page_size: int = 500

class GaiaClient:
    def __init__(self, cfg: GaiaClientConfig):
        self.cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.base_url,
            timeout=cfg.timeout_seconds,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def _get(self, path: str, params: dict) -> dict:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def paginate(self, path: str, params: dict):
        """Yields records across all pages using the API's cursor envelope."""
        cursor = None
        while True:
            page_params = {**params, "page_size": self.cfg.max_page_size}
            if cursor:
                page_params["cursor"] = cursor
            payload = self._get(path, page_params)
            yield from payload["data"]
            if not payload["meta"].get("has_more"):
                break
            cursor = payload["meta"]["next_cursor"]
```

### 4.3 Extractor pattern (example: hourly POS sale, Item 5)

```python
# gaia_pipeline/extractors/possale_hourly.py
import pyarrow as pa
from datetime import date
from gaia_pipeline.gaia_client import GaiaClient
from gaia_pipeline.writer import write_iceberg_append
from gaia_pipeline.watermark import get_watermark, set_watermark

TABLE = "bronze.gaia.possale_hourly"
SCHEMA = pa.schema([
    ("sale_date", pa.date32()),
    ("sale_hour", pa.int8()),
    ("site_code", pa.string()),
    ("tank_id", pa.string()),
    ("product_code", pa.string()),
    ("total_hourly_sale_volume", pa.decimal128(12, 2)),
    ("_ingested_at", pa.timestamp("us")),
    ("_source_batch_id", pa.string()),
])

def extract_and_land(client: GaiaClient, run_date: date, batch_id: str) -> int:
    records = list(client.paginate(
        "/api/v1/possale/hourly",
        params={"date": run_date.isoformat()},
    ))
    if not records:
        return 0

    table = pa.Table.from_pylist(records, schema=SCHEMA.remove_metadata())
    table = table.append_column("_ingested_at", pa.array([...]))   # now(), broadcast
    table = table.append_column("_source_batch_id", pa.array([batch_id] * table.num_rows))

    write_iceberg_append(TABLE, table)
    set_watermark(TABLE, run_date)
    return table.num_rows
```

Every extractor follows the same shape: **pull → validate against schema → tag with batch id and ingest timestamp → append to Iceberg → advance watermark**. This makes each task idempotent — re-running a failed day re-pulls the same `date` param and Iceberg's append is safe to repeat with a fresh `_source_batch_id`, and a `MERGE`-based dedup dbt model in staging drops duplicate batches by `(natural key, _ingested_at desc)`.

### 4.4 Iceberg writer

```python
# gaia_pipeline/writer.py
from pyiceberg.catalog import load_catalog

def _catalog():
    return load_catalog(
        "gaia_lakehouse",
        **{
            "type": "rest",
            "uri": "http://iceberg-rest-catalog:8181",
            "s3.endpoint": "http://minio:9000",
            "s3.access-key-id": "{{ env.MINIO_ACCESS_KEY }}",
            "s3.secret-access-key": "{{ env.MINIO_SECRET_KEY }}",
        },
    )

def write_iceberg_append(table_name: str, arrow_table) -> None:
    catalog = _catalog()
    table = catalog.load_table(table_name)
    table.append(arrow_table)
```

`pyiceberg` (not Spark) is enough here — Airflow tasks are I/O-bound REST pulls landing modest daily volumes; no need for a Spark cluster just to append Parquet into Iceberg. Bring in Spark or Trino-side batch writes only if a single day's extract regularly exceeds a few GB.

### 4.5 Master data extractor (change-triggered, Item 0)

Same client/writer pattern, but calls `?since=<last_watermark>` and writes `change_type` through untouched — Silver's dbt model applies SCD2 based on that field rather than the extractor doing versioning itself. Keeps the Python layer dumb-and-idempotent; all history logic lives in dbt where it's testable and version-controlled.

### 4.6 Airflow DAGs

```python
# dags/gaia_transaction_eod_extract.py
from airflow.decorators import dag, task
from pendulum import datetime as pdt

@dag(
    schedule="0 0 * * *",          # midnight, Asia/Bangkok via DAG timezone
    start_date=pdt(2026, 1, 1),
    catchup=False,
    default_args={"retries": 3, "retry_delay": 300},
    tags=["gaia", "bronze", "eod"],
)
def gaia_transaction_eod_extract():
    @task
    def extract_order_product(ds=None): ...
    @task
    def extract_order_site_tank(ds=None): ...
    @task
    def extract_truck_allocation(ds=None): ...
    @task
    def extract_possale_daily(ds=None): ...
    @task
    def extract_possale_hourly(ds=None): ...
    @task
    def extract_atg_hourly(ds=None): ...

    @task
    def trigger_dbt_bronze_to_silver(): ...   # e.g. dbt Cloud job trigger, or BashOperator `dbt run --select staging`

    (
        [extract_order_product(), extract_order_site_tank(), extract_truck_allocation(),
         extract_possale_daily(), extract_possale_hourly(), extract_atg_hourly()]
        >> trigger_dbt_bronze_to_silver()
    )

gaia_transaction_eod_extract()
```

```python
# dags/gaia_master_data_sync.py
from airflow.decorators import dag, task
from airflow.sensors.base import PokeReturnValue
from pendulum import datetime as pdt

@dag(
    schedule="*/30 * * * *",       # poll every 30 min; swap for a Dataset/webhook trigger once GAIA exposes one
    start_date=pdt(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["gaia", "bronze", "master-data"],
)
def gaia_master_data_sync():
    @task.sensor(poke_interval=60, timeout=1500, mode="reschedule")
    def has_changes() -> PokeReturnValue:
        changed = ...  # HEAD/lightweight call to /api/v1/master/sites?since=<watermark>&count_only=true
        return PokeReturnValue(is_done=changed > 0, xcom_value=changed)

    @task
    def extract_master_sites(): ...

    @task
    def trigger_dbt_master_scd2(): ...

    has_changes() >> extract_master_sites() >> trigger_dbt_master_scd2()

gaia_master_data_sync()
```

### 4.7 Failure handling / SLA
- `retries=3`, exponential backoff at the HTTP client level *and* Airflow task level (don't rely on just one).
- SLA on each DAG: transactions must land by **02:00** (2-hour buffer after midnight trigger); alert to Slack/Teams on breach.
- Data-quality gate task after extraction, before triggering dbt: row-count sanity check against the prior 7-day average per table — hard-fail the DAG (don't promote to Silver) if a table returns zero rows or a >50% drop.

---

## 5. dbt configuration

### 5.1 `dbt_project.yml`

```yaml
name: gaia_lakehouse
version: "1.0.0"
profile: gaia_lakehouse

model-paths: ["models"]
test-paths: ["tests"]

models:
  gaia_lakehouse:
    staging:
      +materialized: view
      +schema: bronze_stg
    intermediate:
      +materialized: ephemeral
    marts:
      silver:
        +materialized: incremental
        +incremental_strategy: merge
        +schema: silver
      gold:
        +materialized: incremental
        +incremental_strategy: merge
        +schema: gold
```

### 5.2 `profiles.yml` (Trino adapter, targeting the Iceberg catalog)

```yaml
gaia_lakehouse:
  target: prod
  outputs:
    prod:
      type: trino
      method: ldap                 # or 'none' / 'kerberos' depending on Trino auth setup
      user: "{{ env_var('DBT_TRINO_USER') }}"
      password: "{{ env_var('DBT_TRINO_PASSWORD') }}"
      host: trino.internal
      port: 443
      http_scheme: https
      catalog: iceberg
      schema: bronze
      threads: 8
```

### 5.3 Sources — `models/staging/_gaia__sources.yml`

```yaml
version: 2
sources:
  - name: gaia
    database: iceberg
    schema: bronze
    tables:
      - name: possale_hourly
        loaded_at_field: _ingested_at
        freshness:
          warn_after: {count: 26, period: hour}
          error_after: {count: 30, period: hour}
      - name: order_site_tank
      - name: master_sites
```

### 5.4 Staging model — `models/staging/stg_gaia__possale_hourly.sql`

```sql
-- 1:1 with Bronze, rename/cast only, dedupe repeated ingest batches
with source as (
    select * from {{ source('gaia', 'possale_hourly') }}
),
deduped as (
    select *,
        row_number() over (
            partition by sale_date, sale_hour, site_code, tank_id, product_code
            order by _ingested_at desc
        ) as rn
    from source
)
select
    sale_date,
    sale_hour,
    site_code,
    tank_id,
    product_code,
    total_hourly_sale_volume
from deduped
where rn = 1
```

### 5.5 Intermediate model — anomaly flags from the checklist

```sql
-- models/intermediate/int_tank_daily_flags.sql
with daily as (
    select * from {{ ref('stg_gaia__order_site_tank_daily') }}
)
select
    *,
    (open_inventory = 0)                          as flag_inven_zero,
    (average_sale_30d = 0)                          as flag_avg_sale_zero,
    (tank_capacity = 0)                             as flag_cap_zero,
    (open_inventory > tank_capacity)                 as flag_inven_gt_capacity,
    (delivery_volume > tank_capacity)                 as flag_delivery_gt_capacity,
    (product_code in ('B10','PREMIUM','E85','GSH98_95_PREMIUM')) as flag_discontinued_product,
    (site_code = 'I150')                              as flag_discontinued_site
from daily
```

### 5.6 Gold feature model — Decision Model features

```sql
-- models/marts/gold/gold_decision_model_features.sql
{{ config(
    materialized='incremental',
    unique_key=['site_code','tank_id','feature_date'],
    incremental_strategy='merge'
) }}

with base as (
    select * from {{ ref('int_tank_daily_flags') }}
    {% if is_incremental() %}
    where feature_date > (select max(feature_date) from {{ this }})
    {% endif %}
)
select
    site_code,
    tank_id,
    feature_date,
    datediff('day', last_delivery_date, feature_date) / nullif(hist_rate_90d, 0) as overdue_ratio,
    (open_inventory + delivery_volume_lag1 - 0.2 * tank_capacity) / nullif(avg_sale_7d, 0) as inv_days_cover,
    avg(delivery_occurred) over (
        partition by site_code, tank_id, day_of_week
        order by feature_date rows between 8 preceding and 1 preceding
    ) as dow_hist_rate,
    avg(delivery_occurred) over (
        partition by site_code, tank_id
        order by feature_date rows between 90 preceding and 1 preceding
    ) as hist_rate,
    sum(delivery_volume) over (
        partition by site_code, tank_id
        order by feature_date rows between 28 preceding and 1 preceding
    ) as delivery_sum_28d,
    (open_inventory + delivery_volume_lag1 - avg_sale_7d) / nullif(tank_capacity, 0) as proj_end_fill_ratio,
    percent_rank() over (partition by feature_date order by overdue_ratio) as group_overdue_pct,
    percent_rank() over (partition by feature_date order by inv_days_cover) as group_cover_pct
from base
```

Volume Model features (`pair_deliv_mean/median/std`, `yoy_deliv`, seasonal flags) follow the same pattern in `gold_volume_model_features.sql` — grouped by `(site_code, product_code)` per the original spec, using `holidays.Thailand` calendar joins for Songkran/New Year flags.

### 5.7 Tests — `models/marts/gold/_gold__schema.yml`

```yaml
version: 2
models:
  - name: gold_decision_model_features
    columns:
      - name: site_code
        tests: [not_null]
      - name: tank_id
        tests: [not_null]
      - name: feature_date
        tests: [not_null]
      - name: overdue_ratio
        tests:
          - dbt_utils.accepted_range:
              min_value: 0
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns: [site_code, tank_id, feature_date]
```

---

## 6. Other tool configuration

### 6.1 MinIO
- Bucket layout: `s3://lakehouse/bronze/`, `s3://lakehouse/silver/`, `s3://lakehouse/gold/` — one bucket per zone (simplifies lifecycle/retention policy and IAM scoping vs one shared bucket).
- Lifecycle rule: Bronze retains raw files 400 days (covers a year of backfill + buffer); Silver/Gold retained indefinitely, managed by Iceberg snapshot expiry instead of bucket lifecycle (`expire_snapshots` on a weekly maintenance DAG).
- Service account per consumer (Airflow writer, dbt/Trino reader-writer, read-only for Superset/Power BI paths if they ever bypass Trino).

### 6.2 Iceberg REST catalog (docker-compose snippet for local/dev)

```yaml
services:
  iceberg-rest:
    image: tabulario/iceberg-rest
    environment:
      CATALOG_WAREHOUSE: s3://lakehouse/
      CATALOG_S3_ENDPOINT: http://minio:9000
      CATALOG_S3_ACCESS_KEY_ID: ${MINIO_ACCESS_KEY}
      CATALOG_S3_SECRET_ACCESS_KEY: ${MINIO_SECRET_KEY}
    ports: ["8181:8181"]
```

### 6.3 Trino Iceberg catalog — `etc/catalog/iceberg.properties`

```properties
connector.name=iceberg
iceberg.catalog.type=rest
iceberg.rest-catalog.uri=http://iceberg-rest:8181
iceberg.rest-catalog.warehouse=s3://lakehouse/
fs.native-s3.enabled=true
s3.endpoint=http://minio:9000
s3.aws-access-key=${ENV:MINIO_ACCESS_KEY}
s3.aws-secret-key=${ENV:MINIO_SECRET_KEY}
s3.path-style-access=true
```

### 6.4 Superset database connection
SQLAlchemy URI: `trino://<user>@trino.internal:443/iceberg/gold?protocol=https`. Set query cost limits (`SUPERSET_ROW_LIMIT`) and enable async query execution for dashboards hitting large Gold marts.

### 6.5 Power BI
Install the Simba Trino ODBC driver, DSN pointing at `trino.internal:443`, catalog `iceberg`, schema `gold`. Use DirectQuery mode for near-real-time dashboards; Import mode only for small reference tables (e.g. site master) refreshed on a schedule.

### 6.6 Airflow connections / variables

| Key | Type | Purpose |
|---|---|---|
| `gaia_api` | HTTP Connection, token in Airflow Secrets backend | GAIA REST auth |
| `minio_s3` | AWS/S3 Connection | pyiceberg S3 credentials |
| `iceberg_rest_catalog` | Generic Connection | catalog URI |
| `trino_default` | Trino Connection | triggering dbt / DQ checks post-load |
| `dq_alert_slack` | Slack webhook Connection | DAG failure / DQ gate alerts |

Secrets backend: Vault (or Airflow's own Fernet-encrypted metadata DB if Vault isn't yet available) — never plaintext in DAG code or `.env` committed to the repo.

### 6.7 Monitoring
- Airflow SLA misses and task failures → Slack channel via `on_failure_callback`.
- dbt: `dbt build --select state:modified+` on CI for PRs; `dbt source freshness` run before each EOD DAG's dbt-trigger task, failing the DAG if Bronze staleness exceeds the freshness thresholds in §5.3.
- Weekly Iceberg maintenance DAG: `expire_snapshots`, `rewrite_data_files` (compaction) to control small-file count from daily appends.

---

## 7. Non-functional requirements
- **PDPA**: mask/tokenize any customer-identifying field before Silver if GAIA's order data ever includes one (not present in Items 0–6 as currently scoped — re-check if fields are added later, per the deck's own open question).
- **Performance target**: EOD DAG completes within the 02:00 SLA for a full day of all six transactional feeds at current site/tank counts.
- **Backfill**: initial 3-year historical load runs as a one-off backfill DAG (separate from the daily incremental DAG), chunked by month to stay within GAIA's per-request capacity limits (see open question below).

## 8. Open questions (carried from GAIA team's own "Next Step" slide)
- Max records per single API pull, and whether GAIA can sustain the 3-year historical backfill in one negotiated window or needs chunked/rate-limited access.
- Typical API response time under load — sizes the timeout/retry config in §4.2.
- Exact hours during which GAIA can serve automated pulls without contention with production traffic.
- Tank cardinality per site — needed to finalize the `bucket(tank_id, N)` parameter in §3.
