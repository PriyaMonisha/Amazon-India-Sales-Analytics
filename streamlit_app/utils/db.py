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
            SELECT p.subcategory AS category
            FROM fact_transactions f
            JOIN dim_products p ON f.product_id = p.product_id
            WHERE f.order_date > '1900-01-01' AND p.subcategory IS NOT NULL
            GROUP BY p.subcategory
            ORDER BY COUNT(*) DESC
            LIMIT 8
        ),
        pairs AS (
            SELECT p1.subcategory AS cat_a, p2.subcategory AS cat_b, COUNT(*) AS co_count
            FROM fact_transactions f1
            JOIN fact_transactions f2
              ON f1.transaction_id = f2.transaction_id AND f1.product_id < f2.product_id
            JOIN dim_products p1 ON f1.product_id = p1.product_id
            JOIN dim_products p2 ON f2.product_id = p2.product_id
            WHERE p1.subcategory IN (SELECT category FROM top8)
              AND p2.subcategory IN (SELECT category FROM top8)
            GROUP BY p1.subcategory, p2.subcategory
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


# ---------------------------------------------------------------------------
# Festival & Seasonal
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_festival_comparison() -> pd.DataFrame:
    return _query("""
        SELECT festival_name,
               COUNT(*)                     AS orders,
               SUM(final_amount_inr)/1e6    AS revenue_m,
               AVG(final_amount_inr)        AS avg_order_value,
               AVG(discount_percent)        AS avg_discount
        FROM fact_transactions
        WHERE is_festival_sale = TRUE
          AND festival_name IS NOT NULL
          AND order_date > '1900-01-01'
        GROUP BY festival_name
        ORDER BY revenue_m DESC
    """)


@st.cache_data(ttl=300)
def get_festival_vs_regular() -> pd.DataFrame:
    return _query("""
        SELECT order_year,
               SUM(CASE WHEN is_festival_sale THEN final_amount_inr ELSE 0 END)/1e6 AS festival_rev_m,
               SUM(CASE WHEN NOT is_festival_sale THEN final_amount_inr ELSE 0 END)/1e6 AS regular_rev_m
        FROM fact_transactions
        WHERE order_date > '1900-01-01' AND order_year BETWEEN 2015 AND 2025
        GROUP BY order_year
        ORDER BY order_year
    """)


@st.cache_data(ttl=300)
def get_monthly_avg_revenue() -> pd.DataFrame:
    return _query("""
        SELECT order_month,
               AVG(monthly_rev) AS avg_rev_m
        FROM (
            SELECT order_year, order_month,
                   SUM(final_amount_inr)/1e6 AS monthly_rev
            FROM fact_transactions
            WHERE order_date > '1900-01-01'
            GROUP BY order_year, order_month
        ) sub
        GROUP BY order_month
        ORDER BY order_month
    """)


@st.cache_data(ttl=300)
def get_festival_subcategory() -> pd.DataFrame:
    return _query("""
        SELECT p.subcategory, f.festival_name,
               SUM(f.final_amount_inr)/1e6 AS revenue_m
        FROM fact_transactions f
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.is_festival_sale = TRUE
          AND f.festival_name IS NOT NULL
          AND f.order_date > '1900-01-01'
          AND p.subcategory IS NOT NULL
        GROUP BY p.subcategory, f.festival_name
        ORDER BY revenue_m DESC
    """)


# ---------------------------------------------------------------------------
# Prime & Demographics
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_prime_detailed() -> pd.DataFrame:
    return _query("""
        SELECT c.is_prime_member,
               COUNT(DISTINCT f.customer_id) AS customers,
               COUNT(*)                       AS orders,
               SUM(f.final_amount_inr)/1e6   AS revenue_m,
               AVG(f.final_amount_inr)        AS avg_order_value,
               AVG(f.discount_percent)        AS avg_discount
        FROM fact_transactions f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        WHERE f.order_date > '1900-01-01'
        GROUP BY c.is_prime_member
    """)


@st.cache_data(ttl=300)
def get_prime_category_preference() -> pd.DataFrame:
    return _query("""
        SELECT c.is_prime_member, p.subcategory,
               SUM(f.final_amount_inr)/1e6 AS revenue_m
        FROM fact_transactions f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.order_date > '1900-01-01' AND p.subcategory IS NOT NULL
        GROUP BY c.is_prime_member, p.subcategory
        ORDER BY revenue_m DESC
    """)


@st.cache_data(ttl=300)
def get_age_group_spending() -> pd.DataFrame:
    return _query("""
        SELECT c.customer_age_group,
               COUNT(DISTINCT f.customer_id) AS customers,
               COUNT(*)                       AS orders,
               SUM(f.final_amount_inr)/1e6   AS revenue_m,
               AVG(f.final_amount_inr)        AS avg_order_value
        FROM fact_transactions f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        WHERE f.order_date > '1900-01-01'
          AND c.customer_age_group IS NOT NULL
          AND c.customer_age_group != 'nan'
        GROUP BY c.customer_age_group
        ORDER BY avg_order_value DESC
    """)


@st.cache_data(ttl=300)
def get_age_subcategory() -> pd.DataFrame:
    return _query("""
        SELECT c.customer_age_group, p.subcategory,
               SUM(f.final_amount_inr)/1e6 AS revenue_m
        FROM fact_transactions f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.order_date > '1900-01-01'
          AND c.customer_age_group IS NOT NULL
          AND c.customer_age_group != 'nan'
          AND p.subcategory IS NOT NULL
        GROUP BY c.customer_age_group, p.subcategory
    """)


@st.cache_data(ttl=300)
def get_tier_spending() -> pd.DataFrame:
    return _query("""
        SELECT c.customer_tier,
               COUNT(DISTINCT f.customer_id) AS customers,
               AVG(f.final_amount_inr)        AS avg_order_value,
               SUM(f.final_amount_inr)/1e6   AS revenue_m,
               AVG(f.discount_percent)        AS avg_discount
        FROM fact_transactions f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        WHERE f.order_date > '1900-01-01' AND c.customer_tier IS NOT NULL
        GROUP BY c.customer_tier
        ORDER BY avg_order_value DESC
    """)


# ---------------------------------------------------------------------------
# Brand & Products
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_brand_detailed(limit: int = 15) -> pd.DataFrame:
    return _query("""
        SELECT p.brand,
               SUM(f.final_amount_inr)/1e9   AS revenue_bn,
               COUNT(*)                       AS orders,
               AVG(f.product_rating)          AS avg_rating,
               AVG(f.discount_percent)        AS avg_discount,
               COUNT(DISTINCT p.product_id)   AS products
        FROM fact_transactions f
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.order_date > '1900-01-01' AND p.brand IS NOT NULL
        GROUP BY p.brand
        ORDER BY revenue_bn DESC
        LIMIT :limit
    """, {"limit": limit})


@st.cache_data(ttl=300)
def get_product_ratings_detail() -> pd.DataFrame:
    return _query("""
        SELECT p.subcategory,
               ROUND(AVG(f.product_rating)::numeric, 2) AS avg_rating,
               COUNT(*) AS reviews,
               SUM(CASE WHEN f.product_rating >= 4 THEN 1 ELSE 0 END)::float / COUNT(*) * 100 AS pct_high,
               SUM(CASE WHEN f.product_rating < 3  THEN 1 ELSE 0 END)::float / COUNT(*) * 100 AS pct_low
        FROM fact_transactions f
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.product_rating IS NOT NULL AND f.order_date > '1900-01-01'
          AND p.subcategory IS NOT NULL
        GROUP BY p.subcategory
        ORDER BY avg_rating DESC
    """)


@st.cache_data(ttl=300)
def get_launch_year_revenue() -> pd.DataFrame:
    return _query("""
        SELECT p.launch_year, p.subcategory,
               SUM(f.final_amount_inr)/1e6 AS revenue_m,
               COUNT(DISTINCT p.product_id) AS products
        FROM fact_transactions f
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE p.launch_year IS NOT NULL AND p.launch_year BETWEEN 2010 AND 2025
          AND f.order_date > '1900-01-01'
        GROUP BY p.launch_year, p.subcategory
        ORDER BY p.launch_year, revenue_m DESC
    """)


@st.cache_data(ttl=300)
def get_returns_detail() -> pd.DataFrame:
    return _query("""
        SELECT p.subcategory,
               COUNT(*) AS total_orders,
               SUM(CASE WHEN f.return_status = 'Returned' THEN 1 ELSE 0 END) AS returns,
               ROUND(SUM(CASE WHEN f.return_status = 'Returned' THEN 1 ELSE 0 END)::numeric
                     / COUNT(*) * 100, 2) AS return_rate,
               AVG(f.product_rating) AS avg_rating,
               AVG(f.discount_percent) AS avg_discount
        FROM fact_transactions f
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.order_date > '1900-01-01' AND p.subcategory IS NOT NULL
        GROUP BY p.subcategory
        ORDER BY return_rate DESC
    """)


# ---------------------------------------------------------------------------
# Customer Journey
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_purchase_frequency() -> pd.DataFrame:
    return _query("""
        SELECT order_count,
               COUNT(*) AS customers
        FROM (
            SELECT customer_id, COUNT(*) AS order_count
            FROM fact_transactions
            WHERE order_date > '1900-01-01'
            GROUP BY customer_id
        ) sub
        GROUP BY order_count
        ORDER BY order_count
    """)


@st.cache_data(ttl=300)
def get_category_transitions() -> pd.DataFrame:
    return _query("""
        WITH ranked AS (
            SELECT f.customer_id, p.subcategory AS category,
                   ROW_NUMBER() OVER (PARTITION BY f.customer_id ORDER BY f.order_date) AS rn
            FROM fact_transactions f
            JOIN dim_products p ON f.product_id = p.product_id
            WHERE f.order_date > '1900-01-01' AND p.subcategory IS NOT NULL
        )
        SELECT r1.category AS from_cat, r2.category AS to_cat,
               COUNT(*) AS transitions
        FROM ranked r1
        JOIN ranked r2 ON r1.customer_id = r2.customer_id AND r2.rn = r1.rn + 1
        GROUP BY r1.category, r2.category
        ORDER BY transitions DESC
    """)


@st.cache_data(ttl=300)
def get_product_performance_summary() -> pd.DataFrame:
    return _query("""
        SELECT p.subcategory,
               SUM(f.final_amount_inr)/1e6   AS revenue_m,
               COUNT(*)                       AS orders,
               COUNT(DISTINCT f.customer_id)  AS customers,
               AVG(f.product_rating)          AS avg_rating,
               ROUND(SUM(CASE WHEN f.return_status='Returned' THEN 1 ELSE 0 END)::numeric
                     / COUNT(*) * 100, 1)     AS return_rate
        FROM fact_transactions f
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.order_date > '1900-01-01' AND p.subcategory IS NOT NULL
        GROUP BY p.subcategory
        ORDER BY revenue_m DESC
    """)


@st.cache_data(ttl=300)
def get_clv_by_segment() -> pd.DataFrame:
    return _query("""
        WITH customer_stats AS (
            SELECT f.customer_id,
                   SUM(f.final_amount_inr) AS clv,
                   COUNT(*)                AS orders,
                   MAX(f.order_date)       AS last_order
            FROM fact_transactions f
            WHERE f.order_date > '1900-01-01'
            GROUP BY f.customer_id
        )
        SELECT c.customer_tier,
               ROUND(AVG(cs.clv)::numeric, 0)    AS avg_clv,
               ROUND(AVG(cs.orders)::numeric, 1)  AS avg_orders,
               COUNT(*)                           AS customers
        FROM customer_stats cs
        JOIN dim_customers c ON cs.customer_id = c.customer_id
        WHERE c.customer_tier IS NOT NULL
        GROUP BY c.customer_tier
        ORDER BY avg_clv DESC
    """)


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


# ---------------------------------------------------------------------------
# Executive Dashboard / Command Centre
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)   # shorter TTL so "live" feel
def get_current_month_perf() -> pd.DataFrame:
    """Latest month revenue + orders vs same month prior year."""
    return _query("""
        WITH latest AS (
            SELECT MAX(order_year * 100 + order_month) AS ym FROM fact_transactions
            WHERE order_date > '1900-01-01'
        ),
        cur AS (
            SELECT order_year, order_month,
                   SUM(final_amount_inr)/1e6   AS rev_m,
                   COUNT(*)                    AS orders,
                   COUNT(DISTINCT customer_id) AS customers,
                   AVG(final_amount_inr)        AS aov
            FROM fact_transactions, latest
            WHERE order_year * 100 + order_month = ym
              AND order_date > '1900-01-01'
            GROUP BY order_year, order_month
        ),
        prev AS (
            SELECT SUM(final_amount_inr)/1e6   AS rev_m_prev,
                   COUNT(*)                    AS orders_prev
            FROM fact_transactions f, latest l
            WHERE f.order_year * 100 + f.order_month =
                  (l.ym / 100 - 1) * 100 + (l.ym % 100)
              AND f.order_date > '1900-01-01'
        )
        SELECT c.*, p.rev_m_prev, p.orders_prev FROM cur c CROSS JOIN prev p
    """)


@st.cache_data(ttl=300)
def get_yearly_revenue() -> pd.DataFrame:
    return _query("""
        SELECT order_year,
               SUM(final_amount_inr)/1e9              AS rev_bn,
               COUNT(*)                               AS orders,
               COUNT(DISTINCT customer_id)            AS customers,
               AVG(final_amount_inr)                  AS aov,
               SUM(discount_percent * final_amount_inr)
                   / NULLIF(SUM(final_amount_inr),0)  AS avg_discount_weighted
        FROM fact_transactions
        WHERE order_date > '1900-01-01' AND order_year BETWEEN 2015 AND 2025
        GROUP BY order_year
        ORDER BY order_year
    """)


@st.cache_data(ttl=300)
def get_category_market_share() -> pd.DataFrame:
    return _query("""
        SELECT p.subcategory AS category,
               SUM(f.final_amount_inr)/1e9  AS rev_bn,
               COUNT(*)                      AS orders
        FROM fact_transactions f
        JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.order_date > '1900-01-01' AND p.subcategory IS NOT NULL
        GROUP BY p.subcategory
        ORDER BY rev_bn DESC
    """)


@st.cache_data(ttl=300)
def get_geographic_spread() -> pd.DataFrame:
    return _query("""
        SELECT c.customer_state,
               c.customer_tier,
               SUM(f.final_amount_inr)/1e6   AS rev_m,
               COUNT(DISTINCT f.customer_id) AS customers
        FROM fact_transactions f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        WHERE f.order_date > '1900-01-01'
          AND c.customer_state IS NOT NULL
        GROUP BY c.customer_state, c.customer_tier
        ORDER BY rev_m DESC
    """)


@st.cache_data(ttl=300)
def get_new_customer_growth() -> pd.DataFrame:
    """First purchase year for each customer → new acquisition per year."""
    return _query("""
        SELECT order_year AS cohort_year,
               COUNT(*) AS new_customers
        FROM (
            SELECT customer_id, MIN(order_year) AS order_year
            FROM fact_transactions
            WHERE order_date > '1900-01-01'
            GROUP BY customer_id
        ) sub
        WHERE order_year BETWEEN 2015 AND 2025
        GROUP BY order_year
        ORDER BY order_year
    """)


@st.cache_data(ttl=300)
def get_discount_revenue_impact() -> pd.DataFrame:
    """Revenue retained vs discount given, by year."""
    return _query("""
        SELECT order_year,
               SUM(original_price_inr * quantity)       AS gross_rev,
               SUM(final_amount_inr)                    AS net_rev,
               SUM(original_price_inr * quantity
                   - final_amount_inr)                  AS discount_given,
               AVG(discount_percent)                    AS avg_discount_pct
        FROM fact_transactions
        WHERE order_date > '1900-01-01'
          AND order_year BETWEEN 2015 AND 2025
          AND original_price_inr IS NOT NULL
        GROUP BY order_year
        ORDER BY order_year
    """)


@st.cache_data(ttl=60)
def get_kpi_alert_data() -> pd.DataFrame:
    """Single-row KPI snapshot used for threshold alerts."""
    return _query("""
        WITH latest_year AS (
            SELECT MAX(order_year) AS yr FROM fact_transactions WHERE order_date > '1900-01-01'
        ),
        prev_year AS (SELECT yr - 1 AS yr FROM latest_year),
        cur  AS (SELECT SUM(final_amount_inr)/1e6 AS rev_m, COUNT(*) AS orders,
                        AVG(final_amount_inr) AS aov
                 FROM fact_transactions, latest_year
                 WHERE order_year = latest_year.yr AND order_date > '1900-01-01'),
        prev AS (SELECT SUM(final_amount_inr)/1e6 AS rev_m_prev
                 FROM fact_transactions, prev_year
                 WHERE order_year = prev_year.yr AND order_date > '1900-01-01'),
        ret  AS (SELECT ROUND(SUM(CASE WHEN return_status='Returned' THEN 1 ELSE 0 END)
                              ::numeric / COUNT(*) * 100, 2) AS return_rate
                 FROM fact_transactions WHERE order_date > '1900-01-01'),
        del  AS (SELECT ROUND(AVG(delivery_days)::numeric, 1) AS avg_del
                 FROM fact_transactions
                 WHERE delivery_days IS NOT NULL AND delivery_days >= 0
                   AND order_date > '1900-01-01')
        SELECT c.rev_m, c.orders, c.aov,
               p.rev_m_prev,
               ROUND(((c.rev_m - p.rev_m_prev) / NULLIF(p.rev_m_prev, 0) * 100)::numeric, 1) AS yoy_growth_pct,
               r.return_rate,
               d.avg_del
        FROM cur c CROSS JOIN prev p CROSS JOIN ret r CROSS JOIN del d
    """)
