from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

import config


@st.cache_resource
def _engine():
    return create_engine(config.DB_URL, pool_pre_ping=True)


def _query(sql: str, params: dict | None = None) -> pd.DataFrame:
    try:
        with _engine().connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_kpis() -> pd.DataFrame:
    return _query("""
        SELECT
            SUM(final_amount_inr)           AS total_revenue,
            COUNT(*)                        AS total_orders,
            COUNT(DISTINCT customer_id)     AS unique_customers,
            AVG(final_amount_inr)           AS avg_order_value
        FROM fact_transactions
        WHERE order_date > '1900-01-01'
    """)


@st.cache_data(ttl=300)
def get_monthly_revenue() -> pd.DataFrame:
    return _query("""
        SELECT order_year, order_month,
               SUM(final_amount_inr) AS revenue,
               COUNT(*)              AS orders
        FROM fact_transactions
        WHERE order_date > '1900-01-01'
        GROUP BY order_year, order_month
        ORDER BY order_year, order_month
    """)


@st.cache_data(ttl=300)
def get_top_subcategories(limit: int = 10) -> pd.DataFrame:
    return _query("""
        SELECT p.subcategory,
               SUM(f.final_amount_inr) AS revenue,
               COUNT(*)                AS orders
        FROM fact_transactions f
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.order_date > '1900-01-01'
          AND p.subcategory IS NOT NULL
        GROUP BY p.subcategory
        ORDER BY revenue DESC
        LIMIT :limit
    """, {"limit": limit})


@st.cache_data(ttl=300)
def get_top_states(limit: int = 10) -> pd.DataFrame:
    return _query("""
        SELECT c.customer_state,
               SUM(f.final_amount_inr) AS revenue,
               COUNT(*)                AS orders
        FROM fact_transactions f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        WHERE f.order_date > '1900-01-01'
          AND c.customer_state IS NOT NULL
        GROUP BY c.customer_state
        ORDER BY revenue DESC
        LIMIT :limit
    """, {"limit": limit})


@st.cache_data(ttl=300)
def get_payment_methods() -> pd.DataFrame:
    return _query("""
        SELECT payment_method,
               COUNT(*)                AS orders,
               SUM(final_amount_inr)   AS revenue
        FROM fact_transactions
        WHERE order_date > '1900-01-01'
          AND payment_method IS NOT NULL
        GROUP BY payment_method
        ORDER BY orders DESC
    """)


# ---------------------------------------------------------------------------
# Churn / Customer
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_customer_tier_distribution() -> pd.DataFrame:
    return _query("""
        SELECT customer_tier,
               COUNT(*) AS customers
        FROM dim_customers
        WHERE customer_tier IS NOT NULL
        GROUP BY customer_tier
        ORDER BY customers DESC
    """)


@st.cache_data(ttl=300)
def get_recency_distribution() -> pd.DataFrame:
    return _query("""
        SELECT customer_id,
               MAX(order_date) AS last_order_date,
               COUNT(*)        AS order_count,
               SUM(final_amount_inr) AS total_spend
        FROM fact_transactions
        WHERE order_date > '1900-01-01'
        GROUP BY customer_id
    """)


@st.cache_data(ttl=300)
def get_monthly_active_customers() -> pd.DataFrame:
    return _query("""
        SELECT order_year, order_month,
               COUNT(DISTINCT customer_id) AS active_customers
        FROM fact_transactions
        WHERE order_date > '1900-01-01'
        GROUP BY order_year, order_month
        ORDER BY order_year, order_month
    """)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_subcategory_list() -> pd.DataFrame:
    return _query("""
        SELECT DISTINCT subcategory
        FROM dim_products
        WHERE subcategory IS NOT NULL
        ORDER BY subcategory
    """)


@st.cache_data(ttl=300)
def get_subcategory_monthly(subcategory: str) -> pd.DataFrame:
    return _query("""
        SELECT f.order_year, f.order_month,
               SUM(f.final_amount_inr) AS revenue,
               COUNT(*)                AS orders
        FROM fact_transactions f
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.order_date > '1900-01-01'
          AND p.subcategory = :subcat
        GROUP BY f.order_year, f.order_month
        ORDER BY f.order_year, f.order_month
    """, {"subcat": subcategory})


@st.cache_data(ttl=300)
def get_top10_heatmap() -> pd.DataFrame:
    return _query("""
        WITH top10 AS (
            SELECT p.subcategory
            FROM fact_transactions f
            JOIN dim_products p ON f.product_id = p.product_id
            WHERE f.order_date > '1900-01-01' AND p.subcategory IS NOT NULL
            GROUP BY p.subcategory
            ORDER BY SUM(f.final_amount_inr) DESC
            LIMIT 10
        )
        SELECT p.subcategory, f.order_month,
               SUM(f.final_amount_inr) / 1e6 AS revenue_m
        FROM fact_transactions f
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.order_date > '1900-01-01'
          AND p.subcategory IN (SELECT subcategory FROM top10)
        GROUP BY p.subcategory, f.order_month
        ORDER BY p.subcategory, f.order_month
    """)


@st.cache_data(ttl=300)
def get_yoy_growth() -> pd.DataFrame:
    return _query("""
        WITH by_year AS (
            SELECT p.subcategory,
                   SUM(CASE WHEN f.order_year = 2023 THEN f.final_amount_inr ELSE 0 END) AS rev_2023,
                   SUM(CASE WHEN f.order_year = 2024 THEN f.final_amount_inr ELSE 0 END) AS rev_2024
            FROM fact_transactions f
            JOIN dim_products p ON f.product_id = p.product_id
            WHERE f.order_date > '1900-01-01'
              AND p.subcategory IS NOT NULL
              AND f.order_year IN (2023, 2024)
            GROUP BY p.subcategory
        )
        SELECT subcategory, rev_2023, rev_2024,
               (rev_2024 - rev_2023) / rev_2023 * 100 AS growth_pct
        FROM by_year
        WHERE rev_2023 > 0 AND rev_2024 > 0
        ORDER BY growth_pct DESC
        LIMIT 15
    """)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_price_distribution(subcategory: str) -> pd.DataFrame:
    return _query("""
        SELECT f.original_price_inr, f.discount_percent, f.final_amount_inr
        FROM fact_transactions f
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.order_date > '1900-01-01'
          AND p.subcategory = :subcat
          AND f.original_price_inr IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 2000
    """, {"subcat": subcategory})


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_popular_subcategories(limit: int = 15) -> pd.DataFrame:
    return _query("""
        SELECT p.subcategory,
               COUNT(DISTINCT f.transaction_id)     AS order_count,
               COUNT(DISTINCT f.customer_id)        AS customer_count,
               SUM(f.final_amount_inr) / 1e6        AS revenue_m
        FROM fact_transactions f
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.order_date > '1900-01-01'
          AND p.subcategory IS NOT NULL
        GROUP BY p.subcategory
        ORDER BY order_count DESC
        LIMIT :limit
    """, {"limit": limit})


@st.cache_data(ttl=300)
def get_copurchase_matrix() -> pd.DataFrame:
    """Category-level co-purchase counts (same transaction_id)."""
    return _query("""
        WITH top8 AS (
            SELECT p.category
            FROM fact_transactions f
            JOIN dim_products p ON f.product_id = p.product_id
            WHERE f.order_date > '1900-01-01' AND p.category IS NOT NULL
            GROUP BY p.category
            ORDER BY COUNT(*) DESC
            LIMIT 8
        ),
        pairs AS (
            SELECT p1.category AS cat_a, p2.category AS cat_b, COUNT(*) AS co_count
            FROM fact_transactions f1
            JOIN fact_transactions f2
              ON f1.transaction_id = f2.transaction_id AND f1.product_id < f2.product_id
            JOIN dim_products p1 ON f1.product_id = p1.product_id
            JOIN dim_products p2 ON f2.product_id = p2.product_id
            WHERE p1.category IN (SELECT category FROM top8)
              AND p2.category IN (SELECT category FROM top8)
            GROUP BY p1.category, p2.category
        )
        SELECT cat_a, cat_b, co_count FROM pairs
        UNION ALL
        SELECT cat_b, cat_a, co_count FROM pairs
        ORDER BY cat_a, cat_b
    """)


# ---------------------------------------------------------------------------
# Anomaly
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_transaction_sample(n: int = 3000) -> pd.DataFrame:
    return _query("""
        SELECT discount_percent, final_amount_inr, delivery_days, return_status
        FROM fact_transactions
        WHERE order_date > '1900-01-01'
          AND discount_percent IS NOT NULL
          AND delivery_days IS NOT NULL
        ORDER BY RANDOM()
        LIMIT :n
    """, {"n": n})


@st.cache_data(ttl=300)
def get_anomaly_summary() -> pd.DataFrame:
    return _query("""
        SELECT
            ROUND(AVG(discount_percent)::numeric, 2)            AS avg_discount_pct,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
                  (ORDER BY discount_percent)::numeric, 2)      AS p95_discount_pct,
            ROUND(AVG(delivery_days)::numeric, 2)               AS avg_delivery_days,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
                  (ORDER BY delivery_days)::numeric, 2)         AS p95_delivery_days,
            ROUND((SUM(CASE WHEN return_status != 'Not Returned'
                        THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100)::numeric, 2) AS return_rate_pct,
            COUNT(*)                                            AS total_transactions
        FROM fact_transactions
        WHERE order_date > '1900-01-01'
    """)
