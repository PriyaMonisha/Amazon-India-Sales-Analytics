from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.utils import api_client, db

st.set_page_config(page_title="Pricing Analytics", page_icon="💰", layout="wide")
st.title("💰 Pricing Analytics")

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
sub_df = db.get_subcategory_list()
subcategories = sub_df["subcategory"].tolist() if not sub_df.empty else []

if not subcategories:
    st.error("PostgreSQL unavailable.")
    st.stop()

col_l, col_r = st.columns([2, 1])
with col_l:
    selected = st.selectbox("Select Subcategory", subcategories, index=0)
with col_r:
    current_price = st.number_input("Current Price (INR)", min_value=1.0, value=1000.0, step=50.0)

# ---------------------------------------------------------------------------
# Fetch pricing scenarios from FastAPI
# ---------------------------------------------------------------------------
pricing_result = api_client.get_pricing(selected, current_price)

if "error" in pricing_result:
    st.warning(f"Pricing model unavailable: {pricing_result['error']}")
    st.caption("Train models first: `notebooks/03_ml_training.py` with Docker + PostgreSQL.")
    scenarios_df = pd.DataFrame()
else:
    suggestions = pricing_result.get("price_suggestions", [])
    scenarios_df = pd.DataFrame(suggestions)
    scenarios_df["label"] = scenarios_df["delta_pct"].apply(
        lambda d: f"{d:+d}%" if d != 0 else "Current"
    )
    is_festival = pricing_result.get("is_festival_month", False)
    month = pricing_result.get("month", "—")
    st.caption(
        f"Month: **{month}**  ·  Festival month: **{'Yes' if is_festival else 'No'}**  ·  "
        f"Current demand estimate: **{pricing_result.get('current_demand_estimate', '—'):,.0f}**"
    )

# ---------------------------------------------------------------------------
# Chart 1 — Revenue by Price Scenario
# ---------------------------------------------------------------------------
st.subheader("Revenue by Price Scenario")
if not scenarios_df.empty:
    best_idx = scenarios_df["revenue_estimate"].idxmax()
    colors = ["#4CAF50" if i == best_idx else "#FF9900" for i in scenarios_df.index]
    fig = go.Figure(go.Bar(
        x=scenarios_df["label"], y=scenarios_df["revenue_estimate"],
        marker_color=colors,
        text=scenarios_df["revenue_estimate"].apply(lambda v: f"₹{v:,.0f}"),
        textposition="outside",
    ))
    fig.update_layout(
        height=320, margin=dict(t=10, b=10),
        xaxis_title="Price Change", yaxis_title="Revenue Estimate (INR)",
        yaxis_tickformat=".2s",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select a subcategory and price to see scenarios.")

# ---------------------------------------------------------------------------
# Chart 2 — Demand by Price Scenario
# ---------------------------------------------------------------------------
st.subheader("Demand Sensitivity to Price")
if not scenarios_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scenarios_df["price_inr"], y=scenarios_df["demand_estimate"],
        mode="lines+markers",
        line=dict(color="#1976D2", width=2),
        marker=dict(size=10, color="#FF9900"),
        text=scenarios_df["label"],
        hovertemplate="Price: ₹%{x:,.0f}<br>Demand: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(t=10, b=10),
        xaxis_title="Price (INR)", yaxis_title="Demand Estimate",
        xaxis_tickformat=".2s",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 3 — Price-Revenue-Demand Trade-off (Bubble Chart)
# ---------------------------------------------------------------------------
st.subheader("Price–Revenue–Demand Trade-off")
if not scenarios_df.empty:
    fig = px.scatter(
        scenarios_df,
        x="price_inr", y="revenue_estimate",
        size="demand_estimate", color="label",
        text="label",
        labels={
            "price_inr": "Price (INR)",
            "revenue_estimate": "Revenue Estimate (INR)",
            "demand_estimate": "Demand",
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        height=330, margin=dict(t=10, b=10),
        xaxis_tickformat=".2s", yaxis_tickformat=".2s",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 4 — Optimal Price Recommendation
# ---------------------------------------------------------------------------
st.subheader("Optimal Price Recommendation")
if not scenarios_df.empty:
    best_row = scenarios_df.loc[best_idx]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimal Price", f"₹{best_row['price_inr']:,.0f}",
              delta=f"{best_row['delta_pct']:+d}% vs current")
    c2.metric("Max Revenue Est.", f"₹{best_row['revenue_estimate']:,.0f}")
    c3.metric("Demand at Opt. Price", f"{best_row['demand_estimate']:,.0f}")
    c4.metric("Current Price", f"₹{current_price:,.0f}")

    revenue_gain = best_row["revenue_estimate"] - scenarios_df.loc[
        scenarios_df["delta_pct"] == 0, "revenue_estimate"
    ].values[0] if (scenarios_df["delta_pct"] == 0).any() else 0
    if revenue_gain > 0:
        st.success(f"Optimal pricing could increase revenue by ~₹{revenue_gain:,.0f} vs current price.")
    elif revenue_gain < 0:
        st.info(f"Current price appears near-optimal for this subcategory.")

# ---------------------------------------------------------------------------
# Chart 5 — Historical Price Distribution from PostgreSQL
# ---------------------------------------------------------------------------
st.subheader(f"Historical Price Distribution — {selected}")
price_df = db.get_price_distribution(selected)
if not price_df.empty:
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(
            price_df, x="original_price_inr", nbins=50,
            labels={"original_price_inr": "Original Price (INR)"},
            color_discrete_sequence=["#9C27B0"],
            title="Original Price Distribution",
        )
        if current_price > 0:
            fig.add_vline(x=current_price, line_dash="dash", line_color="red",
                          annotation_text="Your price")
        fig.update_layout(height=300, margin=dict(t=30, b=10), xaxis_tickformat=".2s")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = px.histogram(
            price_df, x="discount_percent", nbins=40,
            labels={"discount_percent": "Discount (%)"},
            color_discrete_sequence=["#FF5722"],
            title="Discount % Distribution",
        )
        fig2.update_layout(height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.error("PostgreSQL unavailable or subcategory has no transactions.")
