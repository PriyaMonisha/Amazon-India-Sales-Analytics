from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.utils import api_client, db

st.set_page_config(
    page_title="Amazon India Sales Analytics",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Amazon India Sales Analytics")
st.caption("11 years of e-commerce data (2015–2025) · 1.1M transactions · ML-powered insights")

# ---------------------------------------------------------------------------
# System health banner
# ---------------------------------------------------------------------------
health = api_client.get_health()
if "error" in health:
    st.warning(f"FastAPI offline: {health['error']}  —  DB charts still load from PostgreSQL.")
else:
    loaded = health.get("models_loaded", {})
    all_up = all(loaded.values())
    cols = st.columns(len(loaded) + 1)
    cols[0].metric("API", "✅ Online")
    for i, (name, ok) in enumerate(loaded.items(), 1):
        cols[i].metric(name.capitalize(), "✅ Loaded" if ok else "⚠️ Missing")
    if not all_up:
        st.info("Some models not loaded — run `notebooks/03_ml_training.py` with Docker + PostgreSQL.")

st.divider()

# ---------------------------------------------------------------------------
# Chart 1 — KPI metrics (4 st.metric cards)
# ---------------------------------------------------------------------------
st.subheader("Business Overview")
kpi_df = db.get_kpis()
if not kpi_df.empty:
    row = kpi_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"₹{row['total_revenue'] / 1e9:.2f}B")
    c2.metric("Total Orders", f"{int(row['total_orders']):,}")
    c3.metric("Unique Customers", f"{int(row['unique_customers']):,}")
    c4.metric("Avg Order Value", f"₹{row['avg_order_value']:,.0f}")
else:
    st.error("PostgreSQL unavailable — start Docker and run the ETL pipeline first.")
    st.stop()

# ---------------------------------------------------------------------------
# Chart 2 — Monthly Revenue Trend
# ---------------------------------------------------------------------------
st.subheader("Monthly Revenue Trend (2015–2025)")
rev_df = db.get_monthly_revenue()
if not rev_df.empty:
    rev_df["date"] = pd.to_datetime(
        rev_df["order_year"].astype(str) + "-" + rev_df["order_month"].astype(str).str.zfill(2) + "-01"
    )
    rev_df = rev_df[rev_df["order_year"] > 1900].sort_values("date")
    fig = px.line(
        rev_df, x="date", y="revenue",
        labels={"date": "Month", "revenue": "Revenue (INR)"},
        color_discrete_sequence=["#FF9900"],
    )
    fig.update_layout(height=320, margin=dict(t=10, b=10))
    fig.update_yaxes(tickformat=".2s")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Charts 3 & 4 — Top Subcategories + Top States
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    # Chart 3 — Top 10 Subcategories by Revenue
    st.subheader("Top 10 Subcategories by Revenue")
    sub_df = db.get_top_subcategories(10)
    if not sub_df.empty:
        sub_df["revenue_m"] = sub_df["revenue"] / 1e6
        fig = px.bar(
            sub_df.sort_values("revenue_m"),
            x="revenue_m", y="subcategory",
            orientation="h",
            labels={"revenue_m": "Revenue (₹M)", "subcategory": ""},
            color="revenue_m",
            color_continuous_scale="Oranges",
        )
        fig.update_layout(height=380, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    # Chart 4 — Top 10 States by Revenue
    st.subheader("Top 10 States by Revenue")
    state_df = db.get_top_states(10)
    if not state_df.empty:
        state_df["revenue_m"] = state_df["revenue"] / 1e6
        fig = px.bar(
            state_df.sort_values("revenue_m"),
            x="revenue_m", y="customer_state",
            orientation="h",
            labels={"revenue_m": "Revenue (₹M)", "customer_state": ""},
            color="revenue_m",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=380, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 5 — Payment Method Distribution
# ---------------------------------------------------------------------------
st.subheader("Orders by Payment Method")
pay_df = db.get_payment_methods()
if not pay_df.empty:
    fig = px.pie(
        pay_df, values="orders", names="payment_method",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="outside", textinfo="label+percent")
    fig.update_layout(height=350, margin=dict(t=10, b=10), showlegend=False)
    col_mid, _ = st.columns([1, 1])
    with col_mid:
        st.plotly_chart(fig, use_container_width=True)
