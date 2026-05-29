"""Offline / demo-mode fallback for when PostgreSQL is unavailable."""
from __future__ import annotations
from pathlib import Path
import streamlit as st
import config

_CHARTS = config.ARTIFACTS_DIR / "charts"

# Map page → list of EDA chart filenames most relevant to that page
_PAGE_CHARTS: dict[str, list[str]] = {
    "overview": [
        "01_revenue_trend.png",
        "02_seasonal_heatmap.png",
        "07_geographic_tier.png",
        "04_payment_evolution.png",
    ],
    "churn": [
        "03_rfm_segments.png",
        "14_cohort_retention.png",
        "15_clv_distribution.png",
        "09_tier_behaviour.png",
    ],
    "forecast": [
        "01_revenue_trend.png",
        "02_seasonal_heatmap.png",
        "05b_subcategory_growth.png",
        "20_yoy_subcategory.png",
    ],
    "pricing": [
        "11_price_vs_demand.png",
        "17_discount_effectiveness.png",
        "23_competitive_pricing.png",
        "12_brand_performance.png",
    ],
    "recommendations": [
        "05_subcategory_performance.png",
        "21_customer_journey.png",
        "03_rfm_segments.png",
        "10_age_group.png",
    ],
    "anomaly": [
        "17_discount_effectiveness.png",
        "16_delivery_performance.png",
        "13_return_rate.png",
        "11_price_vs_demand.png",
    ],
    "festival": [
        "08_festival_impact.png",
        "02_seasonal_heatmap.png",
        "05b_subcategory_growth.png",
        "04_payment_evolution.png",
    ],
    "prime": [
        "06_prime_impact.png",
        "10_age_group.png",
        "09_tier_behaviour.png",
        "07_geographic_tier.png",
    ],
    "brand": [
        "12_brand_performance.png",
        "18_product_rating.png",
        "22_product_lifecycle.png",
        "13_return_rate.png",
    ],
    "journey": [
        "21_customer_journey.png",
        "14_cohort_retention.png",
        "15_clv_distribution.png",
        "03_rfm_segments.png",
    ],
    "executive": [
        "01_revenue_trend.png",
        "07_geographic_tier.png",
        "05_subcategory_performance.png",
        "20_yoy_subcategory.png",
    ],
}


def show_offline(page: str = "overview") -> None:
    """Display a friendly offline banner + relevant pre-generated EDA charts."""
    st.warning(
        "🔌 **PostgreSQL offline** — showing pre-generated EDA charts below.\n\n"
        "**To see live interactive charts:** double-click **`start_dashboard.bat`** "
        "in the project folder (starts Docker + PostgreSQL automatically).\n\n"
        "Data loads automatically — no re-import needed after the first-time setup."
    )

    charts = _PAGE_CHARTS.get(page, _PAGE_CHARTS["overview"])
    existing = [_CHARTS / c for c in charts if (_CHARTS / c).exists()]

    if not existing:
        st.caption("Pre-generated charts not found. Run `notebooks/02_eda.py` with PostgreSQL to generate them.")
        return

    st.markdown("#### 📊 Pre-Generated Analysis")
    cols = st.columns(2)
    for i, path in enumerate(existing):
        with cols[i % 2]:
            st.image(str(path), use_column_width=True,
                     caption=path.stem.replace("_", " ").title())
