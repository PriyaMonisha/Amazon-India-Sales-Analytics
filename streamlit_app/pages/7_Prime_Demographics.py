from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.utils import db

st.set_page_config(page_title="Prime & Demographics", page_icon="👑", layout="wide")
st.title("👑 Prime Membership & Demographics Analytics")
st.caption("Prime vs non-Prime behaviour · age group preferences · geographic tier analysis")

AGE_ORDER = ["18-25", "26-35", "36-45", "46-55", "55+"]

# ---------------------------------------------------------------------------
# Chart 1 — Prime vs Non-Prime KPI Cards
# ---------------------------------------------------------------------------
st.subheader("Prime vs Non-Prime: Key Metrics")
prime_df = db.get_prime_detailed()
if not prime_df.empty:
    prime_df["label"] = prime_df["is_prime_member"].map({True: "Prime", False: "Non-Prime"}).fillna("Unknown")
    prime_row   = prime_df[prime_df["label"] == "Prime"]
    regular_row = prime_df[prime_df["label"] == "Non-Prime"]

    def _safe(df, col):
        return float(df[col].iloc[0]) if not df.empty and col in df.columns else 0.0

    col1, col2, col3, col4 = st.columns(4)
    aov_lift = (_safe(prime_row, "avg_order_value") / max(_safe(regular_row, "avg_order_value"), 1) - 1) * 100
    col1.metric("Prime Avg Order Value", f"₹{_safe(prime_row,'avg_order_value'):,.0f}",
                delta=f"+{aov_lift:.1f}% vs Non-Prime")
    col2.metric("Non-Prime Avg Order Value", f"₹{_safe(regular_row,'avg_order_value'):,.0f}")
    col3.metric("Prime Revenue Share",
                f"{_safe(prime_row,'revenue_m') / (prime_df['revenue_m'].sum()) * 100:.1f}%")
    col4.metric("Prime Avg Discount", f"{_safe(prime_row,'avg_discount'):.1f}%",
                delta=f"{_safe(prime_row,'avg_discount') - _safe(regular_row,'avg_discount'):+.1f}% vs Non-Prime")

    # Side-by-side bars
    fig = px.bar(
        prime_df[prime_df["label"].isin(["Prime","Non-Prime"])],
        x="label", y=["avg_order_value", "avg_discount"],
        barmode="group",
        color_discrete_map={"avg_order_value": "#FF9900", "avg_discount": "#232F3E"},
        labels={"value": "Value", "label": "", "variable": "Metric"},
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("🔌 PostgreSQL offline — start Docker to view live charts.")

# ---------------------------------------------------------------------------
# Chart 2 — Prime Category Preferences
# ---------------------------------------------------------------------------
st.subheader("Category Preferences: Prime vs Non-Prime")
pcat_df = db.get_prime_category_preference()
if not pcat_df.empty:
    pcat_df["label"] = pcat_df["is_prime_member"].map({True: "Prime", False: "Non-Prime"}).fillna("Unknown")
    pcat_df = pcat_df[pcat_df["label"].isin(["Prime","Non-Prime"])]
    pivot_cat = pcat_df.pivot_table(index="subcategory", columns="label",
                                    values="revenue_m", aggfunc="sum").fillna(0)
    pivot_cat["prime_share"] = pivot_cat["Prime"] / (pivot_cat["Prime"] + pivot_cat["Non-Prime"]) * 100
    pivot_cat = pivot_cat.sort_values("prime_share", ascending=False).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=pivot_cat["subcategory"], y=pivot_cat.get("Prime", 0),
                         name="Prime", marker_color="#FF9900"))
    fig.add_trace(go.Bar(x=pivot_cat["subcategory"], y=pivot_cat.get("Non-Prime", 0),
                         name="Non-Prime", marker_color="#232F3E"))
    fig.update_layout(barmode="stack", height=360, margin=dict(t=10, b=10),
                      xaxis_tickangle=-30,
                      yaxis_title="Revenue (INR Million)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 3 — Age Group Spending Analysis
# ---------------------------------------------------------------------------
st.subheader("Age Group Spending Analysis")
age_df = db.get_age_group_spending()
if not age_df.empty:
    age_df = age_df[age_df["customer_age_group"].isin(AGE_ORDER)].copy()
    age_df["customer_age_group"] = pd.Categorical(age_df["customer_age_group"],
                                                   categories=AGE_ORDER, ordered=True)
    age_df = age_df.sort_values("customer_age_group")

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(
            age_df, x="customer_age_group", y="avg_order_value",
            color="avg_order_value", color_continuous_scale="Blues",
            text="avg_order_value",
            labels={"customer_age_group": "Age Group", "avg_order_value": "Avg Order Value (INR)"},
        )
        fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig2 = px.pie(
            age_df, values="revenue_m", names="customer_age_group",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_traces(textposition="outside", textinfo="label+percent")
        fig2.update_layout(height=320, margin=dict(t=10, b=10),
                           title="Revenue Share by Age Group")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 4 — Age Group × Subcategory Heatmap
# ---------------------------------------------------------------------------
st.subheader("Shopping Preferences by Age Group")
age_sub_df = db.get_age_subcategory()
if not age_sub_df.empty:
    age_sub_df = age_sub_df[age_sub_df["customer_age_group"].isin(AGE_ORDER)]
    pivot_as = age_sub_df.pivot_table(index="customer_age_group", columns="subcategory",
                                      values="revenue_m", aggfunc="sum").fillna(0)
    pivot_as = pivot_as.reindex([a for a in AGE_ORDER if a in pivot_as.index])
    fig = px.imshow(
        pivot_as,
        color_continuous_scale="Blues",
        labels=dict(x="Subcategory", y="Age Group", color="Revenue (INR M)"),
        aspect="auto", text_auto=".0f",
    )
    fig.update_layout(height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 5 — Customer Tier Spending Profile
# ---------------------------------------------------------------------------
st.subheader("Customer Tier Spending Profile")
tier_df = db.get_tier_spending()
if not tier_df.empty:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            tier_df.sort_values("avg_order_value", ascending=True),
            x="avg_order_value", y="customer_tier", orientation="h",
            color="avg_order_value", color_continuous_scale="Greens",
            text="avg_order_value",
            labels={"avg_order_value": "Avg Order Value (INR)", "customer_tier": ""},
        )
        fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.scatter(
            tier_df, x="customers", y="avg_order_value",
            size="revenue_m", color="customer_tier",
            text="customer_tier",
            labels={"customers": "Customer Count", "avg_order_value": "Avg Order Value (INR)",
                    "revenue_m": "Revenue (M)", "customer_tier": "Tier"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig2.update_traces(textposition="top center")
        fig2.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
