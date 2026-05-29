from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.utils import api_client, db, offline

st.set_page_config(page_title="Demand Forecast", page_icon="📈", layout="wide")
st.title("📈 Demand Forecasting")

# ---------------------------------------------------------------------------
# Subcategory selector (used by charts 1 & 2)
# ---------------------------------------------------------------------------
sub_df = db.get_subcategory_list()
subcategories = sub_df["subcategory"].tolist() if not sub_df.empty else []

if not subcategories:
    offline.show_offline("forecast")
    st.stop()

col_sel, col_periods = st.columns([3, 1])
with col_sel:
    selected = st.selectbox("Select Subcategory", subcategories, index=0)
with col_periods:
    periods = st.slider("Forecast Months", 1, 12, 6)

# ---------------------------------------------------------------------------
# Chart 1 — Prophet Forecast with Confidence Interval
# ---------------------------------------------------------------------------
st.subheader(f"Demand Forecast — {selected}")
forecast_result = api_client.get_forecast(selected, periods)

if "error" in forecast_result:
    st.warning(f"Forecast unavailable: {forecast_result['error']}")
    st.caption("Train models first: `notebooks/03_ml_training.py` with Docker + PostgreSQL.")
else:
    fc_points = forecast_result["forecast"]
    fc_df = pd.DataFrame(fc_points)
    fc_df["month"] = pd.to_datetime(fc_df["month"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc_df["month"], y=fc_df["yhat_upper"],
        fill=None, mode="lines", line_color="rgba(255,153,0,0.2)", name="Upper CI",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["month"], y=fc_df["yhat_lower"],
        fill="tonexty", mode="lines", line_color="rgba(255,153,0,0.2)",
        fillcolor="rgba(255,153,0,0.15)", name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=fc_df["month"], y=fc_df["yhat"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#FF9900", width=2), marker=dict(size=8),
    ))
    fig.update_layout(
        height=320, margin=dict(t=10, b=10),
        xaxis_title="Month", yaxis_title="Predicted Revenue (INR)",
        yaxis_tickformat=".2s",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    with st.expander("Forecast Details"):
        display_df = fc_df.copy()
        display_df["month"] = display_df["month"].dt.strftime("%b %Y")
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            display_df[col] = display_df[col].apply(lambda v: f"₹{v:,.0f}")
        display_df.columns = ["Month", "Forecast", "Lower Bound", "Upper Bound"]
        st.dataframe(display_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 2 — Historical Monthly Sales for Selected Subcategory
# ---------------------------------------------------------------------------
st.subheader(f"Historical Sales — {selected}")
hist_df = db.get_subcategory_monthly(selected)
if not hist_df.empty:
    hist_df["date"] = pd.to_datetime(
        hist_df["order_year"].astype(str) + "-" +
        hist_df["order_month"].astype(str).str.zfill(2) + "-01"
    )
    hist_df = hist_df[hist_df["order_year"] > 1900].sort_values("date")
    fig = px.bar(
        hist_df, x="date", y="revenue",
        labels={"date": "Month", "revenue": "Revenue (INR)"},
        color_discrete_sequence=["#1976D2"],
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10), yaxis_tickformat=".2s")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 3 — Top 10 Subcategory Monthly Heatmap
# ---------------------------------------------------------------------------
st.subheader("Revenue Heatmap — Top 10 Subcategories × Month")
heat_df = db.get_top10_heatmap()
if not heat_df.empty:
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    heat_df["month_name"] = heat_df["order_month"].map(month_names)
    pivot = heat_df.pivot_table(
        index="subcategory", columns="month_name", values="revenue_m", aggfunc="mean"
    )
    month_order = [month_names[i] for i in range(1, 13) if month_names[i] in pivot.columns]
    pivot = pivot[month_order]
    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="Subcategory", color="Revenue (₹M)"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 4 — YoY Revenue Growth (2023 vs 2024)
# ---------------------------------------------------------------------------
st.subheader("Year-over-Year Growth: Top 15 Subcategories (2023 → 2024)")
yoy_df = db.get_yoy_growth()
if not yoy_df.empty:
    yoy_df = yoy_df.sort_values("growth_pct")
    colors = ["#4CAF50" if g >= 0 else "#F44336" for g in yoy_df["growth_pct"]]
    fig = go.Figure(go.Bar(
        x=yoy_df["growth_pct"], y=yoy_df["subcategory"],
        orientation="h",
        marker_color=colors,
        text=yoy_df["growth_pct"].apply(lambda v: f"{v:+.1f}%"),
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="gray", line_dash="dot")
    fig.update_layout(
        height=420, margin=dict(t=10, b=10),
        xaxis_title="YoY Growth (%)", yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 5 — Forecast Summary Table across multiple subcategories
# ---------------------------------------------------------------------------
st.subheader("Forecast Summary — Top 5 Subcategories (Next 3 Months)")
st.caption("Calls FastAPI for each subcategory — may take a few seconds.")

top5_subs = sub_df["subcategory"].head(5).tolist()
if st.button("Load Forecast Summary"):
    rows = []
    progress = st.progress(0)
    for i, sub in enumerate(top5_subs):
        res = api_client.get_forecast(sub, 3)
        if "error" not in res and res.get("forecast"):
            pt = res["forecast"][0]
            rows.append({
                "Subcategory": sub,
                "Month 1 Forecast": f"₹{pt['yhat']:,.0f}",
                "Lower": f"₹{pt['yhat_lower']:,.0f}",
                "Upper": f"₹{pt['yhat_upper']:,.0f}",
            })
        progress.progress((i + 1) / len(top5_subs))
    progress.empty()
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.warning("No forecast models loaded — train models first.")
