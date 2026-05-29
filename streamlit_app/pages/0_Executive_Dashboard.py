"""
Executive Command Centre
Covers GUVI Dashboard Q2 (real-time monitor + alerts), Q3 (strategic overview),
Q4 (financial performance), Q5 (growth analytics), Q30 (BI command centre).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from streamlit_app.utils import db

st.set_page_config(page_title="Executive Dashboard", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Sidebar — auto-refresh + alert thresholds
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Dashboard Controls")

    refresh_interval = st.selectbox(
        "Auto-refresh interval",
        options=[None, 60, 300],
        format_func=lambda x: "Off (recommended)" if x is None else f"Every {x}s",
        index=0,
    )
    st.caption(f"Last refreshed: **{datetime.now().strftime('%H:%M:%S')}**")
    st.info(
        "**When to use auto-refresh:**\n\n"
        "• **Off** — normal use; data is historical (2015-2025) and only changes when Airflow ETL runs weekly.\n\n"
        "• **Every 60s** — use during live demo or evaluation to show the refresh capability.\n\n"
        "• **Every 5min** — use when monitoring drift or the daily Airflow drift DAG is running."
    )

    st.divider()
    st.subheader("🚨 Alert Thresholds")
    thr_yoy      = st.slider("Min YoY Growth (%)",  -20, 30, 5)
    thr_return   = st.slider("Max Return Rate (%)",   5, 30, 15)
    thr_delivery = st.slider("Max Avg Delivery Days", 3, 14, 7)

# Auto-refresh via JavaScript (non-blocking — browser reloads the tab)
if refresh_interval:
    components.html(
        f"<script>setTimeout(function(){{window.location.reload()}}, {refresh_interval * 1000});</script>",
        height=0,
    )

st.title("📊 Executive Command Centre")
st.caption("Real-time KPI monitoring · strategic overview · financial health · growth analytics · alerts")

# ---------------------------------------------------------------------------
# Section 1 — KPI Alert Banner (Q2: real-time monitor with alerts)
# ---------------------------------------------------------------------------
alert_df = db.get_kpi_alert_data()

if not alert_df.empty:
    row = alert_df.iloc[0]
    yoy   = float(row.get("yoy_growth_pct", 0) or 0)
    ret   = float(row.get("return_rate", 0) or 0)
    deliv = float(row.get("avg_del", 0) or 0)
    aov   = float(row.get("aov", 0) or 0)

    alerts = []
    if yoy < thr_yoy:
        alerts.append(f"⚠️ **YoY revenue growth {yoy:+.1f}%** — below target of {thr_yoy}%")
    if ret > thr_return:
        alerts.append(f"🔴 **Return rate {ret:.1f}%** — above threshold of {thr_return}%")
    if deliv > thr_delivery:
        alerts.append(f"🔴 **Avg delivery {deliv:.1f} days** — above threshold of {thr_delivery} days")

    if alerts:
        st.error("**Active Alerts**\n\n" + "\n\n".join(alerts))
    else:
        st.success(f"✅ All KPIs within thresholds — YoY {yoy:+.1f}% | Return {ret:.1f}% | Delivery {deliv:.1f}d")

    # KPI cards
    st.subheader("Current Performance vs Prior Year")
    rev_m      = float(row.get("rev_m", 0) or 0)
    rev_prev   = float(row.get("rev_m_prev", 0) or 0)
    orders     = int(row.get("orders", 0) or 0)
    rev_delta  = rev_m - rev_prev

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Revenue (Latest Year)", f"₹{rev_m:,.0f}M",
              delta=f"{rev_delta:+,.0f}M vs prev year")
    c2.metric("YoY Revenue Growth",    f"{yoy:+.1f}%",
              delta_color="normal" if yoy >= thr_yoy else "inverse")
    c3.metric("Avg Order Value",       f"₹{aov:,.0f}")
    c4.metric("Return Rate",           f"{ret:.1f}%",
              delta=f"threshold {thr_return}%",
              delta_color="off")
    c5.metric("Avg Delivery Days",     f"{deliv:.1f}d",
              delta=f"threshold {thr_delivery}d",
              delta_color="off")
else:
    st.warning("PostgreSQL unavailable — start Docker and run ETL pipeline.")

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Revenue & Growth Trends (Q5: Growth Analytics)
# ---------------------------------------------------------------------------
st.subheader("📈 Revenue & Growth Trends (2015–2025)")

yearly_df = db.get_yearly_revenue()
new_cust_df = db.get_new_customer_growth()

if not yearly_df.empty:
    yearly_df["yoy_pct"] = yearly_df["rev_bn"].pct_change() * 100
    yearly_df["cagr_label"] = ""

    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly_df["order_year"], y=yearly_df["rev_bn"],
            name="Revenue (INR Bn)", marker_color="#FF9900",
            text=yearly_df["rev_bn"].map("{:.2f}B".format),
            textposition="inside", textfont=dict(color="black", size=10),
        ))
        fig.add_trace(go.Scatter(
            x=yearly_df["order_year"], y=yearly_df["yoy_pct"],
            name="YoY Growth %", yaxis="y2",
            line=dict(color="white", width=2, dash="dot"),
            mode="lines+markers",
        ))
        fig.update_layout(
            height=360,
            yaxis=dict(title="Revenue (INR Billion)"),
            yaxis2=dict(title="YoY Growth %", overlaying="y", side="right",
                        ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30, b=10, r=60),
            title="Annual Revenue with YoY Growth Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if not new_cust_df.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=new_cust_df["cohort_year"], y=new_cust_df["new_customers"],
                name="New Customers", marker_color="#4CAF50",
                text=new_cust_df["new_customers"].map("{:,}".format),
                textposition="inside", textfont=dict(color="white", size=10),
            ))
            total_customers = int(new_cust_df["new_customers"].sum())
            fig2.add_hline(
                y=new_cust_df["new_customers"].mean(),
                line_dash="dash", line_color="red",
                annotation_text=f"Avg: {new_cust_df['new_customers'].mean():,.0f}",
            )
            fig2.update_layout(
                height=360, margin=dict(t=30, b=10),
                title=f"New Customer Acquisition by Year<br>(Total base: {total_customers:,})",
                yaxis_title="New Customers",
            )
            st.plotly_chart(fig2, use_container_width=True)

    # CAGR summary
    if len(yearly_df) >= 2:
        first_rev = float(yearly_df["rev_bn"].iloc[0])
        last_rev  = float(yearly_df["rev_bn"].iloc[-1])
        n_years   = len(yearly_df) - 1
        cagr = ((last_rev / first_rev) ** (1 / n_years) - 1) * 100 if first_rev > 0 else 0
        first_cust = int(yearly_df["customers"].iloc[0])
        last_cust  = int(yearly_df["customers"].iloc[-1])
        cust_cagr = ((last_cust / first_cust) ** (1 / n_years) - 1) * 100 if first_cust > 0 else 0

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Revenue CAGR (10yr)", f"{cagr:.1f}%")
        cc2.metric("Customer Base CAGR",  f"{cust_cagr:.1f}%")
        cc3.metric("Peak Revenue Year",   str(int(yearly_df.loc[yearly_df["rev_bn"].idxmax(), "order_year"])))
        cc4.metric("Peak Active Customers", f"{yearly_df['customers'].max():,}")

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Strategic Overview: Market Share + Geography (Q3)
# ---------------------------------------------------------------------------
st.subheader("🗺️ Strategic Overview — Market Share & Geographic Spread")

cat_df  = db.get_category_market_share()
geo_df  = db.get_geographic_spread()

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("**Category Market Share**")
    if not cat_df.empty:
        fig = px.pie(
            cat_df, values="rev_bn", names="category",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="inside", textinfo="percent",
                          textfont_size=12)
        fig.update_layout(height=340, margin=dict(t=10, b=80),
                          showlegend=True,
                          legend=dict(orientation="h", yanchor="top",
                                      y=-0.05, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown("**Geographic Revenue Distribution (Top 15 States)**")
    if not geo_df.empty:
        state_agg = geo_df.groupby("customer_state")["rev_m"].sum().nlargest(15).reset_index()
        fig2 = px.bar(
            state_agg.sort_values("rev_m", ascending=True),
            x="rev_m", y="customer_state", orientation="h",
            color="rev_m", color_continuous_scale="Blues",
            labels={"rev_m": "Revenue (INR M)", "customer_state": ""},
            text="rev_m",
        )
        fig2.update_traces(texttemplate="%{text:,.0f}M", textposition="inside",
                           insidetextanchor="end", textfont=dict(size=10))
        fig2.update_layout(height=340, margin=dict(t=10, b=10, r=20),
                           xaxis=dict(range=[0, state_agg["rev_m"].max() * 1.15]))
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

# Tier market penetration
if not geo_df.empty:
    tier_agg = geo_df.groupby("customer_tier").agg(
        rev_m=("rev_m", "sum"), customers=("customers", "sum")
    ).reset_index()
    total_rev = tier_agg["rev_m"].sum()
    total_cust = tier_agg["customers"].sum()
    tier_agg["rev_share"]  = tier_agg["rev_m"]    / total_rev  * 100
    tier_agg["cust_share"] = tier_agg["customers"] / total_cust * 100

    cols = st.columns(len(tier_agg))
    for col, (_, tr) in zip(cols, tier_agg.iterrows()):
        col.metric(
            tr["customer_tier"],
            f"₹{tr['rev_m']:,.0f}M",
            delta=f"{tr['rev_share']:.1f}% revenue | {tr['cust_share']:.1f}% customers",
            delta_color="off",
        )

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Financial Performance (Q4)
# ---------------------------------------------------------------------------
st.subheader("💰 Financial Performance Analysis")

disc_df = db.get_discount_revenue_impact()

if not disc_df.empty:
    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=disc_df["order_year"], y=disc_df["net_rev"] / 1e9,
            name="Net Revenue (after discount)", marker_color="#4CAF50",
        ))
        fig.add_trace(go.Bar(
            x=disc_df["order_year"], y=disc_df["discount_given"] / 1e9,
            name="Discount Given", marker_color="#F44336",
        ))
        fig.update_layout(
            barmode="stack", height=340, margin=dict(t=30, b=10),
            title="Gross Revenue vs Discount Given (INR Billion)",
            yaxis_title="INR Billion",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=disc_df["order_year"], y=disc_df["avg_discount_pct"],
            mode="lines+markers+text",
            line=dict(color="#FF9900", width=2),
            text=disc_df["avg_discount_pct"].map("{:.1f}%".format),
            textposition="top center",
            name="Avg Discount %",
        ))
        disc_df["net_margin_pct"] = disc_df["net_rev"] / disc_df["gross_rev"] * 100
        fig2.add_trace(go.Scatter(
            x=disc_df["order_year"], y=disc_df["net_margin_pct"],
            mode="lines+markers",
            line=dict(color="#4CAF50", width=2, dash="dot"),
            name="Revenue Retention %",
            yaxis="y2",
        ))
        fig2.update_layout(
            height=340, margin=dict(t=30, b=10),
            title="Avg Discount % vs Revenue Retention",
            yaxis=dict(title="Avg Discount (%)", ticksuffix="%"),
            yaxis2=dict(title="Revenue Retention (%)", overlaying="y", side="right",
                        ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Financial summary row
    total_gross   = disc_df["gross_rev"].sum() / 1e9
    total_net     = disc_df["net_rev"].sum()   / 1e9
    total_disc    = disc_df["discount_given"].sum() / 1e9
    avg_retention = (total_net / total_gross * 100) if total_gross else 0

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total Gross Revenue",      f"₹{total_gross:.1f}B")
    f2.metric("Total Net Revenue",        f"₹{total_net:.1f}B")
    f3.metric("Total Discounts Given",    f"₹{total_disc:.1f}B", delta_color="inverse",
              delta=f"-{total_disc/total_gross*100:.1f}% of gross")
    f4.metric("Avg Revenue Retention",    f"{avg_retention:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Section 5 — BI Command Centre Summary (Q30)
# ---------------------------------------------------------------------------
st.subheader("🖥️ BI Command Centre — Business Health Scorecard")

health_items = []

if not yearly_df.empty and not yearly_df["yoy_pct"].dropna().empty:
    last_yoy = float(yearly_df["yoy_pct"].dropna().iloc[-1])
    health_items.append(("Revenue Growth", f"{last_yoy:+.1f}% YoY", "🟢" if last_yoy >= 5 else "🟡" if last_yoy >= 0 else "🔴"))

if not alert_df.empty:
    health_items.append(("Return Rate",    f"{ret:.1f}%",    "🟢" if ret < 10 else "🟡" if ret < 15 else "🔴"))
    health_items.append(("Avg Delivery",   f"{deliv:.1f}d",  "🟢" if deliv <= 5 else "🟡" if deliv <= 7 else "🔴"))
    health_items.append(("Avg Order Value",f"₹{aov:,.0f}",   "🟢"))

if not new_cust_df.empty:
    last2 = new_cust_df.tail(2)["new_customers"].values
    cust_trend = "🟢 Growing" if len(last2) == 2 and last2[1] > last2[0] else "🟡 Flat/Declining"
    health_items.append(("Customer Acquisition", cust_trend, "🟢" if "Growing" in cust_trend else "🟡"))

if not disc_df.empty:
    disc_trend_val = float(disc_df["avg_discount_pct"].iloc[-1])
    health_items.append(("Discount Strategy",
                          f"{disc_trend_val:.1f}% avg discount",
                          "🟢" if disc_trend_val < 20 else "🟡" if disc_trend_val < 30 else "🔴"))

if health_items:
    n_cols = min(len(health_items), 5)
    cols = st.columns(n_cols)
    for i, (label, value, status) in enumerate(health_items):
        cols[i % n_cols].metric(f"{status} {label}", value)

st.divider()

# Navigation quick links — clickable HTML anchors (works in Streamlit 1.35+)
st.markdown("### 🔗 Quick Navigation")
st.caption("Click any card to jump to that analysis page.")

nav_items = [
    ("👥", "Customer Churn",    "/Customer_Churn",    "XGBoost churn model + SHAP explanations"),
    ("📈", "Demand Forecast",   "/Demand_Forecast",   "Prophet forecasting per subcategory"),
    ("💲", "Pricing",           "/Pricing_Analytics", "Price elasticity + revenue optimization"),
    ("🎯", "Recommendations",   "/Recommendations",   "FP-Growth association rules"),
    ("⚠️", "Anomaly Detection", "/Anomaly_Detection", "IsolationForest transaction scoring"),
    ("🎉", "Festival & Season", "/Festival_Seasonal", "Festival impact + seasonal planning"),
    ("👑", "Prime & Demographics","/Prime_Demographics","Prime vs non-Prime + age groups"),
    ("🏷️", "Brand & Products",  "/Brand_Products",    "Brand analytics + ratings + returns"),
    ("🗺️", "Customer Journey",  "/Customer_Journey",  "Transitions + CLV + product scorecard"),
]

# 3 columns × 3 rows grid
for row_start in range(0, len(nav_items), 3):
    row_items = nav_items[row_start:row_start + 3]
    cols = st.columns(3)
    for col, (icon, label, path, desc) in zip(cols, row_items):
        col.markdown(
            f"""<a href="{path}" target="_self" style="text-decoration:none;">
            <div style="border:1px solid #444;border-radius:8px;padding:12px 14px;
                        background:#1e1e2e;cursor:pointer;transition:background 0.2s;"
                 onmouseover="this.style.background='#2d2d44'"
                 onmouseout="this.style.background='#1e1e2e'">
                <span style="font-size:22px">{icon}</span>
                <span style="font-weight:600;font-size:14px;margin-left:8px;color:#FF9900">{label}</span>
                <p style="margin:4px 0 0 0;font-size:12px;color:#aaa">{desc}</p>
            </div></a>""",
            unsafe_allow_html=True,
        )
