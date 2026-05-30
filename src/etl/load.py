"""
Load cleaned DataFrames into PostgreSQL star schema.

Tables:
  fact_transactions  — 1.1M transaction rows
  dim_customers      — 354K unique customers
  dim_products       — 2,004 products
  dim_time           — one row per calendar date

FK note: fact_transactions.order_date has NO FK constraint on dim_time.
         NaT rows are mapped to '1900-01-01' sentinel in transform.py.
         Validation is handled by Great Expectations, not DB constraints.
"""
import logging

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Schema DDL
# ──────────────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dim_time (
    date_id         DATE PRIMARY KEY,
    day             SMALLINT,
    month           SMALLINT,
    quarter         SMALLINT,
    year            SMALLINT,
    is_weekend      BOOLEAN,
    is_festival     BOOLEAN,
    festival_name   TEXT
);

-- Sentinel row for NaT/invalid dates
INSERT INTO dim_time (date_id, day, month, quarter, year, is_weekend, is_festival, festival_name)
VALUES ('1900-01-01', 1, 1, 1, 1900, FALSE, FALSE, 'Unknown')
ON CONFLICT (date_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS dim_customers (
    customer_id            TEXT PRIMARY KEY,
    customer_city          TEXT,
    customer_state         TEXT,
    customer_tier          TEXT,
    customer_age_group     TEXT,
    customer_spending_tier TEXT,
    is_prime_member        BOOLEAN
);

CREATE TABLE IF NOT EXISTS dim_products (
    product_id        TEXT PRIMARY KEY,
    product_name      TEXT,
    category          TEXT,
    subcategory       TEXT,
    brand             TEXT,
    base_price_2015   FLOAT,
    weight_kg         FLOAT,
    rating            FLOAT,
    is_prime_eligible BOOLEAN,
    launch_year       SMALLINT,
    model             TEXT
);

CREATE TABLE IF NOT EXISTS fact_transactions (
    transaction_id       TEXT PRIMARY KEY,
    customer_id          TEXT,
    product_id           TEXT,
    order_date           DATE,
    original_price_inr   FLOAT,
    discount_percent     FLOAT,
    discounted_price_inr FLOAT,
    quantity             SMALLINT,
    subtotal_inr         FLOAT,
    final_amount_inr     FLOAT,
    payment_method       TEXT,
    payment_category     TEXT,
    delivery_days        FLOAT,
    delivery_type        TEXT,
    return_status        TEXT,
    customer_rating      FLOAT,
    rating_missing       SMALLINT,
    is_festival_sale     BOOLEAN,
    festival_name        TEXT,
    sale_type            TEXT,
    order_month          SMALLINT,
    order_year           SMALLINT,
    order_quarter        SMALLINT,
    product_weight_kg    FLOAT,
    is_prime_eligible    BOOLEAN,
    product_rating       FLOAT
);

CREATE INDEX IF NOT EXISTS idx_fact_order_date   ON fact_transactions(order_date);
CREATE INDEX IF NOT EXISTS idx_fact_customer_id  ON fact_transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_fact_product_id   ON fact_transactions(product_id);
CREATE INDEX IF NOT EXISTS idx_fact_date_cust    ON fact_transactions(order_date, customer_id);
CREATE INDEX IF NOT EXISTS idx_fact_year         ON fact_transactions(order_year);
CREATE INDEX IF NOT EXISTS idx_fact_subcategory  ON fact_transactions(order_year);
"""


def get_engine(db_url: str) -> Engine:
    return create_engine(db_url, pool_pre_ping=True)


def create_tables(engine: Engine) -> None:
    with engine.begin() as conn:
        for statement in _SCHEMA_SQL.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
    logger.info("Star schema tables created / verified.")


def _upsert_dataframe(
    df: pd.DataFrame,
    table_name: str,
    primary_key: str,
    engine: Engine,
    chunk_size: int = 10_000,
) -> int:
    """
    Idempotent upsert: INSERT ... ON CONFLICT (pk) DO UPDATE.
    Running ETL twice gives identical result in the database.

    Returns total rows upserted.
    """
    if df.empty:
        logger.warning(f"No data to upsert into {table_name}.")
        return 0

    cols = list(df.columns)
    col_names = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    updates = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols if c != primary_key)

    upsert_sql = f"""
        INSERT INTO {table_name} ({col_names})
        VALUES ({placeholders})
        ON CONFLICT ("{primary_key}") DO UPDATE SET {updates}
    """

    total = 0
    with engine.begin() as conn:
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            records = chunk.to_dict(orient="records")
            conn.execute(text(upsert_sql), records)
            total += len(records)
            logger.debug(f"  {table_name}: upserted chunk {start}–{start+len(records)}")

    logger.info(f"Upserted {total:,} rows into {table_name}.")
    return total


def load_dimensions(
    df_sales: pd.DataFrame,
    df_products: pd.DataFrame,
    engine: Engine,
) -> None:
    """Populate dim_customers, dim_products, dim_time from cleaned data."""

    # dim_customers
    customer_cols = [
        "customer_id", "customer_city", "customer_state", "customer_tier",
        "customer_age_group", "customer_spending_tier", "is_prime_member"
    ]
    available_cust = [c for c in customer_cols if c in df_sales.columns]
    df_customers = (
        df_sales[available_cust]
        .drop_duplicates(subset=["customer_id"])
        .dropna(subset=["customer_id"])
    )
    _upsert_dataframe(df_customers, "dim_customers", "customer_id", engine)

    # dim_products
    product_cols = [
        "product_id", "product_name", "category", "subcategory", "brand",
        "base_price_2015", "weight_kg", "rating", "is_prime_eligible",
        "launch_year", "model"
    ]
    available_prod = [c for c in product_cols if c in df_products.columns]
    df_prod = df_products[available_prod].drop_duplicates(subset=["product_id"])
    _upsert_dataframe(df_prod, "dim_products", "product_id", engine)

    # dim_time (one row per unique date in the dataset)
    if "order_date" in df_sales.columns:
        dates = pd.to_datetime(df_sales["order_date"].dropna().unique())
        # Exclude sentinel
        dates = dates[dates.year > 1900]
        df_time = pd.DataFrame({
            "date_id": dates,
            "day": dates.day,
            "month": dates.month,
            "quarter": dates.quarter,
            "year": dates.year,
            "is_weekend": dates.dayofweek >= 5,
            "is_festival": False,   # updated via festival_name join if needed
            "festival_name": None,
        })
        _upsert_dataframe(df_time, "dim_time", "date_id", engine)


def load_facts(
    df_sales: pd.DataFrame,
    engine: Engine,
    chunk_size: int = 10_000,
) -> int:
    """Load fact_transactions with upsert."""
    fact_cols = [
        "transaction_id", "customer_id", "product_id", "order_date",
        "original_price_inr", "discount_percent", "discounted_price_inr",
        "quantity", "subtotal_inr", "final_amount_inr",
        "payment_method", "payment_category",
        "delivery_days", "delivery_type",
        "return_status", "customer_rating", "rating_missing",
        "is_festival_sale", "festival_name", "sale_type",
        "order_month", "order_year", "order_quarter",
        "product_weight_kg", "is_prime_eligible", "product_rating",
    ]
    available = [c for c in fact_cols if c in df_sales.columns]
    df_fact = df_sales[available].copy()
    # Ensure order_date is date type for PostgreSQL
    if "order_date" in df_fact.columns:
        df_fact["order_date"] = pd.to_datetime(df_fact["order_date"]).dt.date

    return _upsert_dataframe(df_fact, "fact_transactions", "transaction_id", engine, chunk_size)
