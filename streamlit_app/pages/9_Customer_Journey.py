from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.utils import db

st.set_page_config(page_title="Customer Journey", page_icon="🗺️", layout="wide")
st.title("🗺️ Customer Journey & Product Performance")
st.caption("Purchase frequency · category transitions · CLV by segment · full product scorecard")

# ---------------------------------------------------------------------------
# Chart 1 — Purchase Frequency Distribution
# ---------------------------------------------------------------------------
st.subheader("Purchase Frequency Distribution")
freq_df = db.get_purchase_frequency()
if not freq_df.empty:
    # Bucket into readable groups
    def _bucket(n):
        if n == 1:   return "1 order"
        if n == 2:   return "2 orders"
        if n <= 5:   return "3–5 orders"
        if n <= 10:  return "6–10 orders"
        if n <= 20:  return "11–20 orders"
        return "21+ orders"

    BUCKET_ORDER = ["1 order","2 orders","3–5 orders","6–10 orders","11–20 orders","21+ orders"]
    freq_df["bucket"] = freq_df["order_count"].apply(_bucket)
    bucketed = (freq_df.groupby("bucket")["customers"].sum()
                .reindex(BUCKET_ORDER).fillna(0).reset_index())
    bucketed["pct"] = bucketed["customers"] / bucketed["customers"].sum() * 100

    col_l, col_r = st.columns([2, 1])
    with col_l:
        fig = px.bar(
            bucketed, x="bucket", y="customers",
            color="pct", color_continuous_scale="Blues",
            text="customers",
            labels={"bucket": "Orders per Customer", "customers": "Number of Customers", "pct": "Share (%)"},
        )
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        one_time = bucketed[bucketed["bucket"] == "1 order"]["pct"].sum()
        repeat   = 100 - one_time
        loyal    = bucketed[bucketed["bucket"].isin(["11–20 orders","21+ orders"])]["pct"].sum()
        st.metric("One-time Buyers", f"{one_time:.1f}%", delta="opportunity to convert",
                  delta_color="off")
        st.metric("Repeat Customers", f"{repeat:.1f}%")
        st.metric("Loyal Customers (11+ orders)", f"{loyal:.1f}%")
        st.info("**Strategy:** Target one-time buyers with personalised re-engagement within 30 days "
                "to increase repeat purchase rate.")
else:
    st.info("🔌 PostgreSQL offline — start Docker to view live charts.")

# ---------------------------------------------------------------------------
# Chart 2 — Category Transition Heatmap
# ---------------------------------------------------------------------------
st.subheader("Category Transition Matrix")
st.caption("What category do customers buy NEXT after each category? (row → column)")
trans_df = db.get_category_transitions()
if not trans_df.empty:
    pivot_t = trans_df.pivot_table(index="from_cat", columns="to_cat",
                                   values="transitions", aggfunc="sum").fillna(0)
    row_totals = pivot_t.sum(axis=1)
    pivot_pct = pivot_t.div(row_totals, axis=0) * 100

    fig = px.imshow(
        pivot_pct,
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=".0f",
        labels=dict(x="Next Purchase Category", y="Current Category", color="Transition %"),
    )
    fig.update_layout(height=400, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Top transitions table
    top_trans = trans_df.nlargest(8, "transitions")[["from_cat","to_cat","transitions"]].copy()
    top_trans.columns = ["From Category","To Category","Transitions"]
    st.caption("Top 8 most common purchase transitions:")
    st.dataframe(top_trans.reset_index(drop=True), use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 3 — CLV by Customer Segment
# ---------------------------------------------------------------------------
st.subheader("Customer Lifetime Value by Segment")
clv_df = db.get_clv_by_segment()
if not clv_df.empty:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            clv_df.sort_values("avg_clv", ascending=True),
            x="avg_clv", y="customer_tier", orientation="h",
            color="avg_clv", color_continuous_scale="Oranges",
            text="avg_clv",
            labels={"avg_clv": "Avg CLV (INR)", "customer_tier": "Customer Tier"},
        )
        fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.scatter(
            clv_df,
            x="avg_orders", y="avg_clv",
            size="customers", color="customer_tier",
            text="customer_tier",
            labels={"avg_orders": "Avg Orders per Customer", "avg_clv": "Avg CLV (INR)",
                    "customers": "Customer Count", "customer_tier": "Tier"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig2.update_traces(textposition="top center")
        fig2.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False,
                           title="Orders vs CLV by Tier<br>(bubble = number of customers)")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 4 — Full Product Scorecard
# ---------------------------------------------------------------------------
st.subheader("Product Performance Scorecard")
st.caption("Complete subcategory scorecard: revenue · orders · customers · rating · return rate")
prod_df = db.get_product_performance_summary()
if not prod_df.empty:
    # Colour-coded table using plotly
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Subcategory","Revenue (INR M)","Orders","Customers","Avg Rating","Return Rate %"],
            fill_color="#232F3E",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[
                prod_df["subcategory"],
                prod_df["revenue_m"].map("{:,.1f}".format),
                prod_df["orders"].map("{:,}".format),
                prod_df["customers"].map("{:,}".format),
                prod_df["avg_rating"].map("{:.2f}".format),
                prod_df["return_rate"].map("{:.1f}%".format),
            ],
            fill_color=[
                ["white"] * len(prod_df),
                ["#FFF3CD" if v < prod_df["revenue_m"].median() else "#D4EDDA"
                 for v in prod_df["revenue_m"]],
                ["white"] * len(prod_df),
                ["white"] * len(prod_df),
                ["#F8D7DA" if v < 3.5 else "#D4EDDA" for v in prod_df["avg_rating"].fillna(0)],
                ["#F8D7DA" if v > 10 else "#D4EDDA" for v in prod_df["return_rate"].fillna(0)],
            ],
            font=dict(size=11),
            align=["left","right","right","right","center","center"],
        ),
    )])
    fig.update_layout(height=400, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 Green = good  |  🟡 Yellow = below median revenue  |  🔴 Red = low rating or high return rate")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Performing", prod_df.loc[prod_df["revenue_m"].idxmax(), "subcategory"])
    col2.metric("Highest Rated",   prod_df.loc[prod_df["avg_rating"].idxmax(), "subcategory"])
    col3.metric("Lowest Return Rate", prod_df.loc[prod_df["return_rate"].idxmin(), "subcategory"])
