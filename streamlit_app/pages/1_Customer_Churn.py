from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from streamlit_app.utils import api_client, db

st.set_page_config(page_title="Customer Churn", page_icon="👥", layout="wide")
st.title("👥 Customer Churn Analysis")

_BASELINE = config.ARTIFACTS_DIR / "models" / "baseline_churn_proba.json"


def _load_baseline() -> dict | None:
    if _BASELINE.exists():
        return json.loads(_BASELINE.read_text())
    return None


baseline = _load_baseline()

if baseline is None:
    st.info(
        "Baseline not found — train models first: "
        "`notebooks/03_ml_training.py` with Docker + PostgreSQL running."
    )

# ---------------------------------------------------------------------------
# Chart 1 — Churn Probability Distribution
# ---------------------------------------------------------------------------
st.subheader("Churn Probability Distribution")
if baseline:
    probs = baseline["probabilities"]
    threshold = baseline.get("threshold", 0.5)
    churn_rate = baseline.get("churn_rate", None)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=probs, nbinsx=40, name="All customers",
        marker_color="#FF9900", opacity=0.8,
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold {threshold:.2f}", annotation_position="top right")
    fig.update_layout(
        height=300, margin=dict(t=10, b=10),
        xaxis_title="Churn Probability", yaxis_title="Customer Count",
    )
    st.plotly_chart(fig, use_container_width=True)
    if churn_rate is not None:
        st.caption(f"Test-set churn rate: **{churn_rate:.1%}**  ·  n = {baseline.get('n_samples', len(probs)):,} samples")
else:
    st.info("Train models to see this chart.")

# ---------------------------------------------------------------------------
# Chart 2 — Confusion Matrix
# ---------------------------------------------------------------------------
st.subheader("Confusion Matrix (Test Set)")
if baseline:
    labels = np.array(baseline["labels"])
    preds = (np.array(probs) >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    cm = [[tn, fp], [fn, tp]]
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["No Churn", "Churn"], y=["No Churn", "Churn"],
        text_auto=True, color_continuous_scale="Oranges",
    )
    fig.update_layout(height=280, margin=dict(t=10, b=10))
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{precision:.1%}")
    col2.metric("Recall", f"{recall:.1%}")
    col3.metric("F1", f"{2*precision*recall/(precision+recall):.1%}" if (precision + recall) else "N/A")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Train models to see this chart.")

# ---------------------------------------------------------------------------
# Chart 3 — Feature Distributions by Churn Status
# ---------------------------------------------------------------------------
st.subheader("Key Feature Distributions by Churn Status")
if baseline and "feature_distributions" in baseline:
    feat_dist = baseline["feature_distributions"]   # col → list
    labels_arr = np.array(baseline["labels"])
    show_features = [f for f in ["recency_days", "total_orders", "total_spend_inr"] if f in feat_dist]
    if show_features:
        tabs = st.tabs(show_features)
        for tab, feat in zip(tabs, show_features):
            with tab:
                vals = np.array(feat_dist[feat])
                df_feat = pd.DataFrame({"value": vals, "churn": labels_arr.astype(str)})
                df_feat["churn"] = df_feat["churn"].map({"0": "No Churn", "1": "Churn"})
                fig = px.box(
                    df_feat, x="churn", y="value",
                    color="churn",
                    color_discrete_map={"No Churn": "#2196F3", "Churn": "#FF5722"},
                    labels={"value": feat, "churn": ""},
                    points=False,
                )
                fig.update_layout(height=280, margin=dict(t=10, b=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature distribution keys not found in baseline.")
else:
    st.info("Train models to see this chart.")

# ---------------------------------------------------------------------------
# Chart 4 — Customer Tier Distribution
# ---------------------------------------------------------------------------
st.subheader("Customer Tier Distribution")
tier_df = db.get_customer_tier_distribution()
if not tier_df.empty:
    col_l, col_r = st.columns([1, 2])
    with col_l:
        fig = px.pie(
            tier_df, values="customers", names="customer_tier",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        mac_df = db.get_monthly_active_customers()
        if not mac_df.empty:
            mac_df["date"] = pd.to_datetime(
                mac_df["order_year"].astype(str) + "-" +
                mac_df["order_month"].astype(str).str.zfill(2) + "-01"
            )
            mac_df = mac_df[mac_df["order_year"] > 1900].sort_values("date")
            fig2 = px.area(
                mac_df, x="date", y="active_customers",
                labels={"date": "Month", "active_customers": "Active Customers"},
                color_discrete_sequence=["#4CAF50"],
                title="Monthly Active Customers",
            )
            fig2.update_layout(height=320, margin=dict(t=30, b=10))
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("🔌 PostgreSQL offline — start Docker to view live charts.")

# ---------------------------------------------------------------------------
# Chart 5 — Live Churn Predictor
# ---------------------------------------------------------------------------
st.subheader("Live Churn Predictor")
st.caption("Requires FastAPI running + customer materialized in Redis (run Feature Store first).")

with st.form("churn_form"):
    customer_id = st.text_input("Customer ID", placeholder="e.g. CUST_000001")
    explain = st.checkbox("Show SHAP explanation", value=True)
    submitted = st.form_submit_button("Predict")

if submitted and customer_id.strip():
    with st.spinner("Calling FastAPI..."):
        if explain:
            result = api_client.get_churn_explain(customer_id.strip())
        else:
            result = api_client.get_churn(customer_id.strip())

    if "error" in result:
        st.error(result["error"])
    else:
        prob = result["churn_probability"]
        predicted = result["churn_predicted"]
        color = "red" if predicted else "green"
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob:.1%}")
        c2.metric("Prediction", "⚠️ CHURN" if predicted else "✅ Retain")
        c3.metric("Threshold", f"{result['threshold']:.2f}")

        if explain and "top_features" in result:
            feats = result["top_features"][:10]
            feat_df = pd.DataFrame(feats).sort_values("shap_value")
            colors = ["#FF5722" if v > 0 else "#2196F3" for v in feat_df["shap_value"]]
            fig = go.Figure(go.Bar(
                x=feat_df["shap_value"], y=feat_df["feature"],
                orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(
                title="SHAP Feature Contributions (red = increases churn risk)",
                height=350, margin=dict(t=40, b=10),
                xaxis_title="SHAP Value",
            )
            st.plotly_chart(fig, use_container_width=True)
