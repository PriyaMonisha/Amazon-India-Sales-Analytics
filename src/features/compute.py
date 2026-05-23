import logging
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

_FEATURE_QUERY = text("""
SELECT
    dc.customer_id,
    CAST(:ref_date AS DATE)                                                AS event_timestamp,
    CAST(:ref_date AS DATE) - MAX(ft.order_date)::date                    AS days_since_last_purchase,
    COUNT(*) FILTER (
        WHERE ft.order_date >= CAST(:ref_date AS DATE) - INTERVAL '90 days'
    )                                                                      AS total_orders_90d,
    COUNT(*)                                                               AS total_orders_all_time,
    AVG(ft.final_amount_inr) FILTER (
        WHERE ft.order_date >= CAST(:ref_date AS DATE) - INTERVAL '180 days'
    )                                                                      AS avg_order_value_last_6m,
    AVG(ft.final_amount_inr)                                               AS avg_order_value_all_time,
    SUM(ft.final_amount_inr)                                               AS total_spend_all_time,
    BOOL_OR(dc.is_prime_member)                                            AS is_prime_member,
    COALESCE(
        MODE() WITHIN GROUP (ORDER BY dp.category), 'Unknown'
    )                                                                      AS preferred_category,
    COUNT(DISTINCT dp.subcategory)                                         AS unique_categories_purchased,
    AVG(CASE WHEN ft.return_status = 'Returned' THEN 1.0 ELSE 0.0 END)   AS return_rate_historical
FROM fact_transactions ft
JOIN dim_customers dc ON ft.customer_id = dc.customer_id
JOIN dim_products  dp ON ft.product_id  = dp.product_id
WHERE ft.order_date < CAST(:ref_date AS DATE)
GROUP BY dc.customer_id
""")


def _compute_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Recency: lower days = better recency = higher score
    df["recency_score"] = pd.cut(
        df["days_since_last_purchase"].rank(pct=True, ascending=False),
        bins=bins, labels=[5, 4, 3, 2, 1], include_lowest=True,
    ).astype(int)
    df["frequency_score"] = pd.cut(
        df["total_orders_all_time"].rank(pct=True, ascending=True),
        bins=bins, labels=[1, 2, 3, 4, 5], include_lowest=True,
    ).astype(int)
    df["monetary_score"] = pd.cut(
        df["total_spend_all_time"].rank(pct=True, ascending=True),
        bins=bins, labels=[1, 2, 3, 4, 5], include_lowest=True,
    ).astype(int)
    return df


def _assign_rfm_segment(df: pd.DataFrame) -> pd.Series:
    r = df["recency_score"]
    f = df["frequency_score"]
    m = df["monetary_score"]
    conditions = [
        (r >= 4) & (f >= 4) & (m >= 4),
        (r >= 3) & (f >= 3) & (m >= 3),
        (r >= 4) & (f <= 2),
        (r <= 2) & (f >= 3) & (m >= 3),
        (r <= 2) & (f <= 2) & (m <= 2),
        (m >= 4) & (f <= 2),
    ]
    choices = ["Champion", "Loyal", "New", "At Risk", "Lost", "High Value Occasional"]
    return pd.Series(np.select(conditions, choices, default="Potential"), index=df.index)


def compute_customer_features(
    engine: Engine,
    reference_date: date | None = None,
) -> pd.DataFrame:
    """
    Computes customer features as of reference_date from PostgreSQL.

    All time windows (90d, 180d) and days_since_last_purchase are computed
    relative to reference_date — never CURRENT_DATE — to prevent time leakage.

    Saves to:
      data/processed/customer_training_features.parquet  (Feast offline store)
      data/processed/customer_training_features_sample.csv  (1000 rows for inspection)

    Args:
        engine: SQLAlchemy engine pointing to amazon_sales database
        reference_date: "as-of" date for feature computation; defaults to 2024-12-31

    Returns:
        DataFrame with 16 columns: customer_id, event_timestamp, 14 features
    """
    if reference_date is None:
        reference_date = date(2024, 12, 31)

    logger.info("Computing customer features as of %s", reference_date)

    with engine.connect() as conn:
        df = pd.read_sql(_FEATURE_QUERY, conn, params={"ref_date": str(reference_date)})

    if df.empty:
        logger.warning("No customer features computed for reference_date=%s", reference_date)
        return df

    # Feast requires timezone-aware UTC — handle both tz-naive and already tz-aware inputs
    ts = pd.to_datetime(df["event_timestamp"])
    df["event_timestamp"] = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")

    # Fill nulls for customers with no 6-month or 90-day history
    df["avg_order_value_last_6m"] = df["avg_order_value_last_6m"].fillna(
        df["avg_order_value_all_time"]
    )
    df["total_orders_90d"] = df["total_orders_90d"].fillna(0).astype(int)

    df = _compute_rfm_scores(df)
    df["rfm_segment"] = _assign_rfm_segment(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DATA_DIR / "customer_training_features.parquet"
    csv_path = PROCESSED_DATA_DIR / "customer_training_features_sample.csv"

    df.to_parquet(parquet_path, index=False)
    df.head(1000).to_csv(csv_path, index=False)

    logger.info("Saved %d customer features → %s", len(df), parquet_path)
    logger.info("Sample CSV (1000 rows) → %s", csv_path)

    return df
