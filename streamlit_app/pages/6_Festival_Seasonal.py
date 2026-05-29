from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.utils import db

st.set_page_config(page_title="Festival & Seasonal", page_icon="🎉", layout="wide")
st.title("🎉 Festival & Seasonal Analytics")
st.caption("Festival sales impact · seasonal revenue patterns · promotional planning")

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------------------------------------------------------------------
# Chart 1 — Festival Revenue Comparison
# ---------------------------------------------------------------------------
st.subheader("Festival Revenue Comparison")
fest_df = db.get_festival_comparison()
if not fest_df.empty:
    col_l, col_r = st.columns([3, 2])
    with col_l:
        fig = px.bar(
            fest_df.sort_values("revenue_m", ascending=True),
            x="revenue_m", y="festival_name", orientation="h",
            color="avg_order_value",
            color_continuous_scale="Oranges",
            labels={"revenue_m": "Revenue (INR Million)", "festival_name": "",
                    "avg_order_value": "Avg Order (INR)"},
            text="revenue_m",
        )
        fig.update_traces(texttemplate="%{text:.0f}M", textposition="outside")
        fig.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig2 = px.bar(
            fest_df.sort_values("avg_discount", ascending=False),
            x="festival_name", y="avg_discount",
            color="avg_discount", color_continuous_scale="Reds",
            labels={"avg_discount": "Avg Discount (%)", "festival_name": ""},
            text="avg_discount",
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(height=350, margin=dict(t=10, b=10), showlegend=False)
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
    with st.expander("Festival Summary Table"):
        display = fest_df.copy()
        display["revenue_m"] = display["revenue_m"].map("{:,.1f}M".format)
        display["avg_order_value"] = display["avg_order_value"].map("₹{:,.0f}".format)
        display["avg_discount"] = display["avg_discount"].map("{:.1f}%".format)
        display.columns = ["Festival", "Orders", "Revenue", "Avg Order Value", "Avg Discount"]
        st.dataframe(display.reset_index(drop=True), use_container_width=True)
else:
    st.error("PostgreSQL unavailable — start Docker and run ETL pipeline.")

# ---------------------------------------------------------------------------
# Chart 2 — Festival vs Regular Sales by Year
# ---------------------------------------------------------------------------
st.subheader("Festival vs Regular Revenue by Year")
fvr_df = db.get_festival_vs_regular()
if not fvr_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fvr_df["order_year"], y=fvr_df["festival_rev_m"],
        name="Festival Sales", marker_color="#FF9900",
    ))
    fig.add_trace(go.Bar(
        x=fvr_df["order_year"], y=fvr_df["regular_rev_m"],
        name="Regular Sales", marker_color="#232F3E",
    ))
    fvr_df["fest_pct"] = fvr_df["festival_rev_m"] / (fvr_df["festival_rev_m"] + fvr_df["regular_rev_m"]) * 100
    fig.add_trace(go.Scatter(
        x=fvr_df["order_year"], y=fvr_df["fest_pct"],
        name="Festival %", yaxis="y2",
        line=dict(color="red", dash="dot", width=2),
        mode="lines+markers",
    ))
    fig.update_layout(
        barmode="stack", height=380,
        yaxis=dict(title="Revenue (INR Million)"),
        yaxis2=dict(title="Festival Share (%)", overlaying="y", side="right",
                    range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 3 — Seasonal Pattern (Monthly Average)
# ---------------------------------------------------------------------------
st.subheader("Seasonal Revenue Pattern")
monthly_df = db.get_monthly_avg_revenue()
if not monthly_df.empty:
    monthly_df["month_name"] = monthly_df["order_month"].apply(lambda m: MONTH_NAMES[int(m)-1])
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            monthly_df, x="month_name", y="avg_rev_m",
            color="avg_rev_m", color_continuous_scale="YlOrRd",
            labels={"avg_rev_m": "Avg Monthly Revenue (INR M)", "month_name": "Month"},
            text="avg_rev_m",
        )
        fig.update_traces(texttemplate="%{text:.0f}M", textposition="outside")
        fig.update_layout(height=340, margin=dict(t=10, b=10), showlegend=False)
        fig.update_coloraxes(showscale=False)
        fig.update_xaxes(categoryorder="array", categoryarray=MONTH_NAMES)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Annotate peak months
        peak_month = monthly_df.loc[monthly_df["avg_rev_m"].idxmax(), "month_name"]
        low_month  = monthly_df.loc[monthly_df["avg_rev_m"].idxmin(), "month_name"]
        st.metric("Peak Month", peak_month, help="Highest average monthly revenue")
        st.metric("Lowest Month", low_month, help="Lowest average monthly revenue")
        total_avg = monthly_df["avg_rev_m"].sum()
        q4_avg = monthly_df[monthly_df["order_month"].isin([10, 11, 12])]["avg_rev_m"].sum()
        st.metric("Q4 Share of Annual Revenue", f"{q4_avg/total_avg*100:.1f}%",
                  help="Oct–Dec share (festival quarter)")
        st.info("**Insight:** Q4 (Oct–Dec) is consistently the highest-revenue quarter, "
                "driven by Diwali, Big Billion Days, and year-end shopping. "
                "Plan inventory and promotions 6–8 weeks ahead.")

# ---------------------------------------------------------------------------
# Chart 4 — Festival Performance by Subcategory
# ---------------------------------------------------------------------------
st.subheader("Top Subcategories by Festival")
fsub_df = db.get_festival_subcategory()
if not fsub_df.empty:
    festivals = ["All"] + sorted(fsub_df["festival_name"].dropna().unique().tolist())
    selected_fest = st.selectbox("Filter by festival", festivals)
    if selected_fest != "All":
        fsub_df = fsub_df[fsub_df["festival_name"] == selected_fest]
    top_fsub = fsub_df.groupby("subcategory")["revenue_m"].sum().nlargest(10).reset_index()
    fig = px.bar(
        top_fsub.sort_values("revenue_m", ascending=True),
        x="revenue_m", y="subcategory", orientation="h",
        color="revenue_m", color_continuous_scale="Oranges",
        labels={"revenue_m": "Revenue (INR M)", "subcategory": ""},
        text="revenue_m",
    )
    fig.update_traces(texttemplate="%{text:.0f}M", textposition="outside")
    fig.update_layout(height=350, margin=dict(t=10, b=10))
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)
