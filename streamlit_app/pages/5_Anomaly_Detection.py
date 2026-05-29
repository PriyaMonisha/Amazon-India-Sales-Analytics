from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.utils import api_client, db

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly Detection")

# ---------------------------------------------------------------------------
# Chart 1 — Live Anomaly Checker Form
# ---------------------------------------------------------------------------
st.subheader("Transaction Anomaly Checker")

with st.form("anomaly_form"):
    st.caption("Enter transaction details to check for anomalous patterns.")
    col1, col2 = st.columns(2)
    with col1:
        final_amount = st.number_input("Final Amount Paid (INR)", min_value=1.0, value=500.0, step=50.0)
        mrp = st.number_input("MRP / Original Price (INR)", min_value=1.0, value=1000.0, step=50.0)
    with col2:
        delivery_days = st.number_input("Delivery Days", min_value=0.0, value=3.0, step=0.5)
        is_return = st.selectbox("Return?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    submitted = st.form_submit_button("Check for Anomaly", type="primary")

# ---------------------------------------------------------------------------
# Chart 2 — Anomaly Score Gauge
# ---------------------------------------------------------------------------
anomaly_result = None
if submitted:
    if final_amount > mrp:
        st.warning("Final amount > MRP — unusual but will still run the check.")
    with st.spinner("Running anomaly detection..."):
        anomaly_result = api_client.post_anomaly(
            final_amount_inr=final_amount,
            mrp_inr=mrp,
            delivery_days=delivery_days,
            is_return=is_return,
        )

if anomaly_result:
    if "error" in anomaly_result:
        st.error(f"Anomaly model unavailable: {anomaly_result['error']}")
        st.caption("Train models first: `notebooks/03_ml_training.py` with Docker + PostgreSQL.")
    else:
        is_anom = anomaly_result["is_anomaly"]
        score = anomaly_result["anomaly_score"]
        disc_pct = anomaly_result["discount_pct"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Verdict", "⚠️ ANOMALY" if is_anom else "✅ Normal",
                  delta="Flagged" if is_anom else "Clean",
                  delta_color="inverse" if is_anom else "normal")
        c2.metric("Anomaly Score", f"{score:.4f}",
                  help="Higher = more normal (IsolationForest decision function)")
        c3.metric("Effective Discount", f"{disc_pct:.1%}")

        # Gauge chart for anomaly score
        score_norm = (score + 1) / 2  # rough normalization for display
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Anomaly Score (higher = more normal)"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "#4CAF50" if not is_anom else "#F44336"},
                "steps": [
                    {"range": [-1, -0.1], "color": "#FFCDD2"},
                    {"range": [-0.1, 0.1], "color": "#FFF9C4"},
                    {"range": [0.1, 1], "color": "#C8E6C9"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": 0,
                },
            },
        ))
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Summary stats from PostgreSQL
# ---------------------------------------------------------------------------
summary_df = db.get_anomaly_summary()
if not summary_df.empty:
    row = summary_df.iloc[0]
else:
    row = None

# ---------------------------------------------------------------------------
# Chart 3 — Historical Discount Distribution
# ---------------------------------------------------------------------------
st.subheader("Historical Discount Distribution")
sample_df = db.get_transaction_sample(3000)
if not sample_df.empty:
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.histogram(
            sample_df, x="discount_percent", nbins=50,
            labels={"discount_percent": "Discount (%)"},
            color_discrete_sequence=["#FF9900"],
        )
        if submitted and anomaly_result and "error" not in anomaly_result:
            checked_discount = (1 - final_amount / mrp) * 100
            fig.add_vline(x=checked_discount, line_dash="dash", line_color="red",
                          annotation_text="Your transaction")
        if row is not None:
            fig.add_vline(x=float(row["p95_discount_pct"]), line_dash="dot", line_color="orange",
                          annotation_text="P95")
        fig.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="Discount (%)")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Chart 4 — Delivery Days Distribution
    # ---------------------------------------------------------------------------
    with col_r:
        fig2 = px.histogram(
            sample_df[sample_df["delivery_days"] < 60], x="delivery_days", nbins=40,
            labels={"delivery_days": "Delivery Days"},
            color_discrete_sequence=["#1976D2"],
        )
        if submitted:
            fig2.add_vline(x=delivery_days, line_dash="dash", line_color="red",
                           annotation_text="Your transaction")
        if row is not None:
            fig2.add_vline(x=float(row["p95_delivery_days"]), line_dash="dot", line_color="orange",
                           annotation_text="P95")
        fig2.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="Delivery Days")
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("🔌 PostgreSQL offline — start Docker to view live charts.")

# ---------------------------------------------------------------------------
# Chart 5 — Transaction Summary Statistics
# ---------------------------------------------------------------------------
st.subheader("Transaction Population Statistics")
if row is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Discount", f"{float(row['avg_discount_pct']):.1f}%")
    c2.metric("P95 Discount", f"{float(row['p95_discount_pct']):.1f}%")
    c3.metric("Avg Delivery Days", f"{float(row['avg_delivery_days']):.1f}")
    c4.metric("Return Rate", f"{float(row['return_rate_pct']):.1f}%")

    # Bar chart of key stats
    stats = {
        "Avg Discount (%)": float(row["avg_discount_pct"]),
        "P95 Discount (%)": float(row["p95_discount_pct"]),
        "Avg Delivery Days": float(row["avg_delivery_days"]),
        "P95 Delivery Days": float(row["p95_delivery_days"]),
        "Return Rate (%)": float(row["return_rate_pct"]),
    }
    stats_df = pd.DataFrame({"Metric": list(stats.keys()), "Value": list(stats.values())})

    if submitted and anomaly_result and "error" not in anomaly_result:
        checked_discount = (1 - final_amount / mrp) * 100
        user_stats = {
            "Avg Discount (%)": checked_discount,
            "P95 Discount (%)": checked_discount,
            "Avg Delivery Days": delivery_days,
            "P95 Delivery Days": delivery_days,
            "Return Rate (%)": float(is_return) * 100,
        }
        user_df = pd.DataFrame({"Metric": list(user_stats.keys()), "Value": list(user_stats.values())})

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stats_df["Metric"], y=stats_df["Value"],
            name="Population", marker_color="#FF9900",
        ))
        fig.add_trace(go.Bar(
            x=user_df["Metric"], y=user_df["Value"],
            name="Your Transaction",
            marker_color="#F44336" if (anomaly_result and anomaly_result.get("is_anomaly")) else "#4CAF50",
        ))
        fig.update_layout(
            barmode="group", height=320, margin=dict(t=10, b=10),
            yaxis_title="Value", xaxis_title="",
        )
    else:
        fig = px.bar(
            stats_df, x="Metric", y="Value",
            color_discrete_sequence=["#FF9900"],
        )
        fig.update_layout(height=300, margin=dict(t=10, b=10))

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Based on {int(row['total_transactions']):,} transactions.")
else:
    st.info("🔌 PostgreSQL offline — start Docker to view live charts.")
