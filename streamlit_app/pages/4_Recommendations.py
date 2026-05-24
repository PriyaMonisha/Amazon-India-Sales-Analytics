from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.utils import api_client, db

st.set_page_config(page_title="Recommendations", page_icon="🛒", layout="wide")
st.title("🛒 Product Recommendations")

# ---------------------------------------------------------------------------
# Subcategory selector
# ---------------------------------------------------------------------------
sub_df = db.get_subcategory_list()
subcategories = sub_df["subcategory"].tolist() if not sub_df.empty else []

if not subcategories:
    st.error("PostgreSQL unavailable.")
    st.stop()

col_l, col_r = st.columns([3, 1])
with col_l:
    selected = st.selectbox("Input Subcategory (basket item)", subcategories, index=0)
with col_r:
    top_n = st.slider("Top N Recommendations", 3, 15, 10)

# ---------------------------------------------------------------------------
# Fetch recommendations from FastAPI
# ---------------------------------------------------------------------------
rec_result = api_client.get_recommendations(selected, top_n)

if "error" in rec_result:
    rec_df = pd.DataFrame()
    st.warning(f"Recommendation model unavailable: {rec_result['error']}")
    st.caption("Train models first: `notebooks/03_ml_training.py` with Docker + PostgreSQL.")
else:
    recs = rec_result.get("recommendations", [])
    rec_df = pd.DataFrame(recs) if recs else pd.DataFrame()

# ---------------------------------------------------------------------------
# Chart 1 — Top Recommendations Bar (Confidence)
# ---------------------------------------------------------------------------
st.subheader(f"Top {top_n} Recommendations for '{selected}'")
if not rec_df.empty:
    rule_df = rec_df[rec_df["source"] == "association_rules"].copy()
    fallback_df = rec_df[rec_df["source"] == "popularity_fallback"].copy()
    display_df = rec_df.copy()
    display_df["confidence_val"] = display_df["confidence"].fillna(0.0)

    colors = [
        "#FF9900" if s == "association_rules" else "#78909C"
        for s in display_df["source"]
    ]
    fig = go.Figure(go.Bar(
        x=display_df["confidence_val"],
        y=display_df["subcategory"],
        orientation="h",
        marker_color=colors,
        text=display_df["confidence_val"].apply(
            lambda v: f"{v:.1%}" if v > 0 else "popular"
        ),
        textposition="outside",
    ))
    fig.update_layout(
        height=max(300, top_n * 32), margin=dict(t=10, b=10),
        xaxis_title="Confidence", yaxis_title="",
        xaxis_range=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Orange = association rule  ·  Gray = popularity fallback")
else:
    st.info("No recommendations returned — model may not be loaded.")

# ---------------------------------------------------------------------------
# Chart 2 — Confidence vs Lift Scatter
# ---------------------------------------------------------------------------
st.subheader("Confidence vs Lift (Association Rules)")
if not rec_df.empty and "association_rules" in rec_df["source"].values:
    rule_df = rec_df[rec_df["source"] == "association_rules"].dropna(subset=["confidence", "lift"])
    if not rule_df.empty:
        fig = px.scatter(
            rule_df, x="confidence", y="lift",
            text="subcategory",
            size_max=15,
            color_discrete_sequence=["#FF9900"],
            labels={"confidence": "Confidence", "lift": "Lift"},
        )
        fig.update_traces(textposition="top center", marker=dict(size=12))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      annotation_text="Lift = 1 (random)", annotation_position="right")
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No association rules with confidence + lift for this subcategory.")
else:
    st.info("Train the FP-Growth model to see association rule statistics.")

# ---------------------------------------------------------------------------
# Chart 3 — Source Breakdown Pie
# ---------------------------------------------------------------------------
if not rec_df.empty:
    col_pie, col_meta = st.columns([1, 2])
    with col_pie:
        st.subheader("Recommendation Source")
        source_counts = rec_df["source"].value_counts().reset_index()
        source_counts.columns = ["Source", "Count"]
        fig = px.pie(
            source_counts, values="Count", names="Source", hole=0.5,
            color_discrete_map={
                "association_rules": "#FF9900",
                "popularity_fallback": "#78909C",
            },
        )
        fig.update_traces(textinfo="label+percent+value")
        fig.update_layout(height=280, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_meta:
        st.subheader("Recommendation Table")
        tbl = rec_df[["subcategory", "confidence", "lift", "source"]].copy()
        tbl["confidence"] = tbl["confidence"].apply(
            lambda v: f"{v:.1%}" if pd.notna(v) and v > 0 else "—"
        )
        tbl["lift"] = tbl["lift"].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) else "—"
        )
        tbl.columns = ["Subcategory", "Confidence", "Lift", "Source"]
        st.dataframe(tbl, hide_index=True, use_container_width=True, height=260)

# ---------------------------------------------------------------------------
# Chart 4 — Category Co-purchase Heatmap (PostgreSQL)
# ---------------------------------------------------------------------------
st.subheader("Category Co-purchase Heatmap (Top 8 Categories)")
st.caption("Items bought together in the same transaction — across all 11 years.")
copurchase_df = db.get_copurchase_matrix()
if not copurchase_df.empty:
    pivot = copurchase_df.pivot_table(
        index="cat_a", columns="cat_b", values="co_count", fill_value=0
    )
    fig = px.imshow(
        pivot,
        labels=dict(x="Category B", y="Category A", color="Co-purchases"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
        text_auto=True,
    )
    fig.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Co-purchase data not available — ensure PostgreSQL is running.")

# ---------------------------------------------------------------------------
# Chart 5 — Subcategory Popularity (PostgreSQL)
# ---------------------------------------------------------------------------
st.subheader("Top 15 Subcategories by Order Volume")
pop_df = db.get_popular_subcategories(15)
if not pop_df.empty:
    pop_df = pop_df.sort_values("order_count")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pop_df["order_count"], y=pop_df["subcategory"],
        orientation="h", name="Orders",
        marker_color="#1976D2",
    ))
    fig.add_trace(go.Bar(
        x=pop_df["customer_count"], y=pop_df["subcategory"],
        orientation="h", name="Unique Customers",
        marker_color="#FF9900",
    ))
    fig.update_layout(
        barmode="overlay", height=440, margin=dict(t=10, b=10),
        xaxis_title="Count", yaxis_title="",
        xaxis_tickformat=".2s",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("PostgreSQL unavailable.")
