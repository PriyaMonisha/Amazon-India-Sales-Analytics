# %% [markdown]
# # Section 2: Exploratory Data Analysis (20 Analyses)
#
# All analyses query from PostgreSQL (not CSV).
# Charts saved to artifacts/charts/ for Streamlit display.
#
# Run locally: `make eda`  (after `make etl`)

# %%
import logging
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()

sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eda_notebook")

# %%
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves files instead of displaying

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.stats import ttest_ind
from sqlalchemy import text

import config
from src.etl.load import get_engine

CHARTS_DIR = config.ARTIFACTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

engine = get_engine(config.DB_URL)
logger.info(f"Connected to {config.DB_URL[:60]}...")

# Discrete color palette for all categorical charts (never sequential on categorical)
PALETTE = px.colors.qualitative.Set2
SNS_PALETTE = "Set2"


def save_chart(fig: plt.Figure, name: str) -> None:
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved chart: {name}")


def query(sql: str, **params) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


# %% [markdown]
# ## Analysis 1 — Revenue Trend 2015–2025

# %%
df_rev = query("""
    SELECT order_year, SUM(final_amount_inr) AS revenue, COUNT(*) AS orders
    FROM fact_transactions
    WHERE order_year BETWEEN 2015 AND 2025
    GROUP BY order_year ORDER BY order_year
""")
df_rev["yoy_growth_pct"] = df_rev["revenue"].pct_change() * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_rev["order_year"], df_rev["revenue"] / 1e9, marker="o", linewidth=2, color=PALETTE[0])
ax.set_title("Amazon India Revenue Trend 2015–2025", fontsize=14, fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Revenue (₹ Billion)")
for _, row in df_rev.iterrows():
    if not pd.isna(row["yoy_growth_pct"]):
        ax.annotate(f"+{row['yoy_growth_pct']:.0f}%", (row["order_year"], row["revenue"] / 1e9),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="green")
plt.tight_layout()
save_chart(fig, "01_revenue_trend.png")

print("Insight: Revenue grew from ₹X B (2015) to ₹Y B (2025). COVID-2020 caused a spike as Indians shifted to online shopping.")

# %% [markdown]
# ## Analysis 2 — Seasonal Heatmap (Year × Month)

# %%
df_heat = query("""
    SELECT order_year, order_month, SUM(final_amount_inr) AS revenue
    FROM fact_transactions
    WHERE order_year BETWEEN 2015 AND 2025
    GROUP BY order_year, order_month
""")
pivot = df_heat.pivot(index="order_year", columns="order_month", values="revenue").fillna(0)
pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot / 1e6, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("Monthly Revenue Heatmap (₹ Million)", fontsize=13, fontweight="bold")
ax.set_ylabel("Year"); ax.set_xlabel("Month")
plt.tight_layout()
save_chart(fig, "02_seasonal_heatmap.png")

print("Insight: October–November (Diwali / Big Billion Days) consistently show the highest revenue across all years.")

# %% [markdown]
# ## Analysis 3 — RFM Customer Segmentation

# %%
df_rfm_raw = query("""
    SELECT customer_id,
           MAX(order_year * 100 + order_month) AS last_period,
           COUNT(*) AS frequency,
           SUM(final_amount_inr) AS monetary
    FROM fact_transactions
    GROUP BY customer_id
""")
# Recency = how recent (higher = more recent)
max_period = df_rfm_raw["last_period"].max()
df_rfm_raw["recency"] = max_period - df_rfm_raw["last_period"]

# Score 1–5 per dimension
for col in ["recency", "frequency", "monetary"]:
    df_rfm_raw[f"{col}_score"] = pd.qcut(df_rfm_raw[col], q=5, labels=[5, 4, 3, 2, 1] if col == "recency" else [1, 2, 3, 4, 5], duplicates="drop")

# Segment labels
def rfm_segment(row: pd.Series) -> str:
    r, f, m = row["recency_score"], row["frequency_score"], row["monetary_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3 and m >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "Promising"
    if r <= 2 and f >= 3:
        return "At Risk"
    return "Hibernating"

df_rfm_raw["segment"] = df_rfm_raw.apply(rfm_segment, axis=1)

segment_counts = df_rfm_raw["segment"].value_counts()
fig = px.pie(values=segment_counts.values, names=segment_counts.index,
             color_discrete_sequence=PALETTE, title="Customer RFM Segmentation")
fig.write_image(str(CHARTS_DIR / "03_rfm_segmentation.png"))
logger.info("Saved chart: 03_rfm_segmentation.png")

print(f"Insight: Champions = {segment_counts.get('Champions', 0):,} | Loyal = {segment_counts.get('Loyal', 0):,} | At Risk = {segment_counts.get('At Risk', 0):,}")

# %% [markdown]
# ## Analysis 4 — Payment Method Evolution 2015–2025

# %%
df_pay = query("""
    SELECT order_year, payment_method, COUNT(*) AS orders
    FROM fact_transactions
    WHERE order_year BETWEEN 2015 AND 2025
    GROUP BY order_year, payment_method
""")
pivot_pay = df_pay.pivot(index="order_year", columns="payment_method", values="orders").fillna(0)
pivot_pct = pivot_pay.div(pivot_pay.sum(axis=1), axis=0) * 100

fig = px.area(pivot_pct.reset_index(), x="order_year", y=pivot_pct.columns.tolist(),
              color_discrete_sequence=PALETTE,
              title="Payment Method Evolution (Market Share %)",
              labels={"value": "Market Share (%)", "order_year": "Year"})
fig.write_image(str(CHARTS_DIR / "04_payment_evolution.png"))
logger.info("Saved chart: 04_payment_evolution.png")

print("Insight: UPI grew from near-zero (2015) to dominant (2023+). COD declined sharply post-2016 demonetization.")

# %% [markdown]
# ## Analysis 5 — Subcategory Performance (Treemap + Bar + Line)

# %%
df_cat = query("""
    SELECT p.subcategory,
           SUM(f.final_amount_inr) AS revenue,
           COUNT(*) AS orders
    FROM fact_transactions f
    JOIN dim_products p ON f.product_id = p.product_id
    GROUP BY p.subcategory
    ORDER BY revenue DESC
""")
df_cat["market_share_pct"] = df_cat["revenue"] / df_cat["revenue"].sum() * 100

fig_tree = px.treemap(df_cat, path=["subcategory"], values="revenue",
                      color="market_share_pct", color_continuous_scale="Blues",
                      title="Revenue Distribution by Subcategory (Treemap)")
fig_tree.write_image(str(CHARTS_DIR / "05a_subcategory_treemap.png"))

fig_bar, ax = plt.subplots(figsize=(10, 5))
colors = [PALETTE[i % len(PALETTE)] for i in range(len(df_cat))]
bars = ax.bar(df_cat["subcategory"], df_cat["revenue"] / 1e9, color=colors)
ax.bar_label(bars, [f"{v:.1f}B" for v in df_cat["revenue"] / 1e9], fontsize=9)
ax.set_title("Revenue by Subcategory (₹ Billion)"); ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save_chart(fig_bar, "05b_subcategory_bar.png")

# Growth line
df_cat_year = query("""
    SELECT order_year, p.subcategory, SUM(f.final_amount_inr) AS revenue
    FROM fact_transactions f
    JOIN dim_products p ON f.product_id = p.product_id
    WHERE order_year BETWEEN 2015 AND 2025
    GROUP BY order_year, p.subcategory
""")
fig_line = px.line(df_cat_year, x="order_year", y="revenue", color="subcategory",
                   markers=True, color_discrete_sequence=PALETTE,
                   title="Subcategory Revenue Growth Over Years")
fig_line.write_image(str(CHARTS_DIR / "05c_subcategory_growth.png"))
logger.info("Saved charts: 05a, 05b, 05c subcategory")

# %% [markdown]
# ## Analysis 6 — Prime Membership Impact

# %%
df_prime = query("""
    SELECT is_prime_member,
           AVG(final_amount_inr) AS aov,
           COUNT(*) AS total_orders,
           COUNT(DISTINCT customer_id) AS unique_customers
    FROM fact_transactions
    GROUP BY is_prime_member
""")
df_prime["orders_per_customer"] = df_prime["total_orders"] / df_prime["unique_customers"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
palette = [PALETTE[0], PALETTE[1]]

# AOV
bars0 = axes[0].bar(df_prime["is_prime_member"].astype(str), df_prime["aov"], color=palette)
axes[0].bar_label(bars0, [f"₹{v:.0f}" for v in df_prime["aov"]]); axes[0].set_title("Average Order Value")

# Order frequency
bars1 = axes[1].bar(df_prime["is_prime_member"].astype(str), df_prime["orders_per_customer"], color=palette)
axes[1].bar_label(bars1, [f"{v:.1f}" for v in df_prime["orders_per_customer"]]); axes[1].set_title("Orders per Customer")

# Total orders
bars2 = axes[2].bar(df_prime["is_prime_member"].astype(str), df_prime["total_orders"], color=palette)
axes[2].bar_label(bars2, [f"{v/1000:.0f}K" for v in df_prime["total_orders"]]); axes[2].set_title("Total Orders")

plt.suptitle("Prime vs Non-Prime Customer Behavior", fontsize=13, fontweight="bold")
plt.tight_layout()
save_chart(fig, "06_prime_membership_impact.png")

# %% [markdown]
# ## Analysis 7 — Geographic Analysis

# %%
df_state = query("""
    SELECT c.customer_state, SUM(f.final_amount_inr) AS revenue, COUNT(*) AS orders
    FROM fact_transactions f
    JOIN dim_customers c ON f.customer_id = c.customer_id
    GROUP BY c.customer_state ORDER BY revenue DESC LIMIT 15
""")
fig, ax = plt.subplots(figsize=(12, 5))
colors = [PALETTE[i % len(PALETTE)] for i in range(len(df_state))]
bars = ax.bar(df_state["customer_state"], df_state["revenue"] / 1e9, color=colors)
ax.bar_label(bars, [f"{v:.1f}B" for v in df_state["revenue"] / 1e9], fontsize=8)
ax.set_title("Top 15 States by Revenue (₹ Billion)"); ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save_chart(fig, "07_geographic_state.png")

df_tier = query("""
    SELECT c.customer_tier, SUM(f.final_amount_inr) AS revenue, COUNT(*) AS orders
    FROM fact_transactions f
    JOIN dim_customers c ON f.customer_id = c.customer_id
    GROUP BY c.customer_tier ORDER BY revenue DESC
""")
fig_tier = px.bar(df_tier, x="customer_tier", y="revenue", color="customer_tier",
                  color_discrete_sequence=PALETTE, title="Revenue by Customer Tier",
                  text_auto=True)
fig_tier.write_image(str(CHARTS_DIR / "07b_tier_revenue.png"))
logger.info("Saved charts: 07_geographic_state, 07b_tier_revenue")

# %% [markdown]
# ## Analysis 8 — Festival Impact (Before / During / After)

# %%
df_festival = query("""
    SELECT order_date, is_festival_sale, festival_name, final_amount_inr
    FROM fact_transactions
    WHERE order_date > '1900-01-01'
""")
df_festival["order_date"] = pd.to_datetime(df_festival["order_date"])

daily_rev = df_festival.groupby("order_date")["final_amount_inr"].sum().reset_index()

# Festival windows: 7 days before / festival period / 7 days after
festival_groups = df_festival[df_festival["is_festival_sale"] == True].groupby("festival_name")["order_date"]
window_data = []
for fest, dates in festival_groups:
    if pd.isna(fest):
        continue
    start, end = dates.min(), dates.max()
    before_rev = daily_rev[(daily_rev["order_date"] >= start - pd.Timedelta(days=7)) &
                           (daily_rev["order_date"] < start)]["final_amount_inr"].sum()
    during_rev = daily_rev[(daily_rev["order_date"] >= start) &
                           (daily_rev["order_date"] <= end)]["final_amount_inr"].sum()
    after_rev = daily_rev[(daily_rev["order_date"] > end) &
                          (daily_rev["order_date"] <= end + pd.Timedelta(days=7))]["final_amount_inr"].sum()
    uplift = (during_rev - before_rev) / before_rev * 100 if before_rev > 0 else 0
    window_data.append({"festival": fest, "before": before_rev, "during": during_rev, "after": after_rev, "uplift_%": uplift})

df_window = pd.DataFrame(window_data)
if not df_window.empty:
    df_melt = df_window.melt(id_vars="festival", value_vars=["before", "during", "after"],
                             var_name="period", value_name="revenue")
    fig_fest = px.bar(df_melt, x="festival", y="revenue", color="period",
                      barmode="group", color_discrete_sequence=PALETTE,
                      title="Festival Revenue: Before / During / After")
    fig_fest.write_image(str(CHARTS_DIR / "08_festival_impact.png"))
    logger.info("Saved chart: 08_festival_impact.png")

# T-test: festival vs non-festival daily revenue
fest_rev = df_festival[df_festival["is_festival_sale"] == True]["final_amount_inr"]
non_fest_rev = df_festival[df_festival["is_festival_sale"] == False]["final_amount_inr"]
t_stat, p_val = ttest_ind(fest_rev, non_fest_rev, equal_var=False)
print(f"T-test: t={t_stat:.2f}, p={p_val:.4e} — {'significant' if p_val < 0.05 else 'not significant'} at α=0.05")

# %% [markdown]
# ## Analysis 9 — Customer Tier Behavior

# %%
df_tier_b = query("""
    SELECT c.customer_tier,
           AVG(f.final_amount_inr) AS avg_order,
           AVG(f.discount_percent) AS avg_discount,
           f.payment_method,
           COUNT(*) AS cnt
    FROM fact_transactions f
    JOIN dim_customers c ON f.customer_id = c.customer_id
    GROUP BY c.customer_tier, f.payment_method
    ORDER BY c.customer_tier, cnt DESC
""")
fig_tier_b = px.box(
    query("SELECT c.customer_tier, f.final_amount_inr FROM fact_transactions f JOIN dim_customers c ON f.customer_id=c.customer_id"),
    x="customer_tier", y="final_amount_inr", color="customer_tier",
    color_discrete_sequence=PALETTE, title="Order Value Distribution by Customer Tier"
)
fig_tier_b.write_image(str(CHARTS_DIR / "09_tier_order_distribution.png"))
logger.info("Saved chart: 09_tier_order_distribution.png")

# %% [markdown]
# ## Analysis 10 — Age Group Preferences

# %%
df_age = query("""
    SELECT c.customer_age_group, p.subcategory,
           SUM(f.final_amount_inr) AS revenue
    FROM fact_transactions f
    JOIN dim_customers c ON f.customer_id = c.customer_id
    JOIN dim_products p ON f.product_id = p.product_id
    GROUP BY c.customer_age_group, p.subcategory
""")
pivot_age = df_age.pivot(index="customer_age_group", columns="subcategory", values="revenue").fillna(0)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot_age / 1e6, annot=True, fmt=".0f", cmap="Blues", ax=ax, linewidths=0.3)
ax.set_title("Revenue (₹M) by Age Group × Subcategory"); ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save_chart(fig, "10_age_group_subcategory.png")

# %% [markdown]
# ## Analysis 11 — Price vs Demand

# %%
df_pd = query("""
    SELECT discounted_price_inr, quantity,
           discount_percent, p.subcategory
    FROM fact_transactions f
    JOIN dim_products p ON f.product_id = p.product_id
    WHERE discounted_price_inr IS NOT NULL AND quantity IS NOT NULL
    LIMIT 50000
""")
fig_sc = px.scatter(df_pd.sample(min(5000, len(df_pd))),
                    x="discounted_price_inr", y="quantity",
                    color="subcategory", color_discrete_sequence=PALETTE,
                    opacity=0.5, title="Price vs Demand by Subcategory")
fig_sc.write_image(str(CHARTS_DIR / "11_price_vs_demand.png"))

# Correlation heatmap
num_cols = ["discounted_price_inr", "original_price_inr", "discount_percent", "quantity"]
available = [c for c in num_cols if c in df_pd.columns]
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df_pd[available].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Price-Demand Correlation")
plt.tight_layout()
save_chart(fig, "11b_price_demand_corr.png")
logger.info("Saved charts: 11_price_vs_demand, 11b_price_demand_corr")

# %% [markdown]
# ## Analysis 12 — Brand Performance

# %%
df_brand = query("""
    SELECT p.brand, SUM(f.final_amount_inr) AS revenue,
           AVG(f.discount_percent) AS avg_discount,
           AVG(f.product_rating) AS avg_rating,
           COUNT(*) AS orders
    FROM fact_transactions f
    JOIN dim_products p ON f.product_id = p.product_id
    GROUP BY p.brand ORDER BY revenue DESC LIMIT 15
""")
fig_br, ax = plt.subplots(figsize=(12, 5))
colors = [PALETTE[i % len(PALETTE)] for i in range(len(df_brand))]
bars = ax.bar(df_brand["brand"], df_brand["revenue"] / 1e9, color=colors)
ax.bar_label(bars, [f"{v:.1f}B" for v in df_brand["revenue"] / 1e9], fontsize=8)
ax.set_title("Top 15 Brands by Revenue (₹ Billion)"); ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save_chart(fig_br, "12_brand_revenue.png")

# Discount vs rating scatter
fig_dr = px.scatter(df_brand, x="avg_discount", y="avg_rating", size="revenue",
                    text="brand", color_discrete_sequence=PALETTE,
                    title="Brand: Avg Discount vs Avg Rating (bubble=revenue)")
fig_dr.write_image(str(CHARTS_DIR / "12b_brand_discount_rating.png"))
logger.info("Saved charts: 12_brand_revenue, 12b_brand_discount_rating")

# %% [markdown]
# ## Analysis 13 — Return Rate Analysis

# %%
df_return = query("""
    SELECT p.subcategory, f.payment_method, c.customer_city,
           SUM(CASE WHEN f.return_status = 'Returned' THEN 1 ELSE 0 END)::float / COUNT(*) AS return_rate,
           COUNT(*) AS orders
    FROM fact_transactions f
    JOIN dim_products p ON f.product_id = p.product_id
    JOIN dim_customers c ON f.customer_id = c.customer_id
    GROUP BY p.subcategory, f.payment_method, c.customer_city
""")
subcat_return = df_return.groupby("subcategory")["return_rate"].mean().reset_index().sort_values("return_rate", ascending=False)
fig_ret, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(subcat_return["subcategory"], subcat_return["return_rate"] * 100,
              color=[PALETTE[i % len(PALETTE)] for i in range(len(subcat_return))])
ax.bar_label(bars, [f"{v:.1f}%" for v in subcat_return["return_rate"] * 100], fontsize=9)
ax.set_title("Return Rate by Subcategory (%)"); ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save_chart(fig_ret, "13_return_rate_subcategory.png")

# %% [markdown]
# ## Analysis 14 — Cohort Retention Matrix

# %%
df_cohort_raw = query("""
    SELECT customer_id, order_year
    FROM fact_transactions
    WHERE order_year BETWEEN 2015 AND 2025
""")
first_purchase = df_cohort_raw.groupby("customer_id")["order_year"].min().reset_index()
first_purchase.columns = ["customer_id", "cohort_year"]
df_cohort = df_cohort_raw.merge(first_purchase, on="customer_id")
cohort_matrix = (
    df_cohort.groupby(["cohort_year", "order_year"])["customer_id"]
    .nunique()
    .unstack(fill_value=0)
)
# Normalize by cohort size (first year = 100%)
cohort_pct = cohort_matrix.divide(cohort_matrix.iloc[:, 0], axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(cohort_pct, annot=True, fmt=".0f", cmap="Blues", ax=ax, linewidths=0.3)
ax.set_title("Customer Cohort Retention (% of Cohort Still Active)"); ax.set_xlabel("Order Year")
plt.tight_layout()
save_chart(fig, "14_cohort_retention.png")

# %% [markdown]
# ## Analysis 15 — Customer Lifetime Value Distribution

# %%
df_clv = query("""
    SELECT customer_id, SUM(final_amount_inr) AS clv
    FROM fact_transactions
    GROUP BY customer_id
""")
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df_clv["clv"].clip(upper=df_clv["clv"].quantile(0.99)) / 1000, bins=50,
        color=PALETTE[0], edgecolor="white", alpha=0.8)
ax.axvline(df_clv["clv"].mean() / 1000, color="red", ls="--", label=f"Mean: ₹{df_clv['clv'].mean()/1000:.1f}K")
ax.axvline(df_clv["clv"].median() / 1000, color="green", ls=":", label=f"Median: ₹{df_clv['clv'].median()/1000:.1f}K")
ax.set_title("Customer Lifetime Value Distribution (₹K)"); ax.set_xlabel("CLV (₹K)"); ax.legend()
plt.tight_layout()
save_chart(fig, "15_clv_distribution.png")

# %% [markdown]
# ## Analysis 16 — Delivery Performance

# %%
df_del = query("""
    SELECT delivery_type, c.customer_tier, delivery_days
    FROM fact_transactions f
    JOIN dim_customers c ON f.customer_id = c.customer_id
    WHERE delivery_days IS NOT NULL AND delivery_days >= 0
""")
fig_del = px.box(df_del, x="delivery_type", y="delivery_days", color="customer_tier",
                 color_discrete_sequence=PALETTE,
                 title="Delivery Days by Delivery Type & Customer Tier")
fig_del.write_image(str(CHARTS_DIR / "16_delivery_performance.png"))
logger.info("Saved chart: 16_delivery_performance.png")

# %% [markdown]
# ## Analysis 17 — Discount Effectiveness

# %%
df_disc = query("""
    SELECT discount_percent, quantity, final_amount_inr
    FROM fact_transactions
    WHERE discount_percent IS NOT NULL
    LIMIT 50000
""")
fig_disc = px.scatter(df_disc.sample(min(5000, len(df_disc))),
                      x="discount_percent", y="quantity",
                      trendline="ols", color_discrete_sequence=PALETTE,
                      title="Discount % vs Quantity Ordered")
fig_disc.write_image(str(CHARTS_DIR / "17_discount_effectiveness.png"))
logger.info("Saved chart: 17_discount_effectiveness.png")

# %% [markdown]
# ## Analysis 18 — Product Rating Distribution

# %%
df_rat = query("""
    SELECT p.subcategory, f.product_rating
    FROM fact_transactions f
    JOIN dim_products p ON f.product_id = p.product_id
    WHERE f.product_rating IS NOT NULL
""")
fig_rat = px.box(df_rat, x="subcategory", y="product_rating", color="subcategory",
                 color_discrete_sequence=PALETTE,
                 title="Product Rating Distribution by Subcategory")
fig_rat.write_image(str(CHARTS_DIR / "18_product_rating_dist.png"))
logger.info("Saved chart: 18_product_rating_dist.png")

# %% [markdown]
# ## Analysis 19 — Revenue Concentration (Pareto)

# %%
df_pareto = query("""
    SELECT p.product_id, SUM(f.final_amount_inr) AS revenue
    FROM fact_transactions f
    JOIN dim_products p ON f.product_id = p.product_id
    GROUP BY p.product_id ORDER BY revenue DESC
""")
df_pareto["cumulative_pct"] = df_pareto["revenue"].cumsum() / df_pareto["revenue"].sum() * 100
df_pareto["product_pct"] = np.arange(1, len(df_pareto) + 1) / len(df_pareto) * 100

threshold_80 = df_pareto[df_pareto["cumulative_pct"] <= 80]["product_pct"].max()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_pareto["product_pct"], df_pareto["cumulative_pct"], color=PALETTE[0], linewidth=2)
ax.axhline(80, color="red", ls="--", alpha=0.7, label="80% revenue")
ax.axvline(threshold_80, color="orange", ls="--", alpha=0.7, label=f"Top {threshold_80:.0f}% products")
ax.set_title("Pareto Chart: Revenue Concentration"); ax.set_xlabel("% Products"); ax.set_ylabel("Cumulative % Revenue")
ax.legend()
plt.tight_layout()
save_chart(fig, "19_revenue_pareto.png")
print(f"Pareto: Top {threshold_80:.0f}% of products generate 80% of revenue.")

# %% [markdown]
# ## Analysis 20 — YoY Growth by Subcategory (Small Multiples)

# %%
df_yoy = query("""
    SELECT order_year, p.subcategory, SUM(f.final_amount_inr) AS revenue
    FROM fact_transactions f
    JOIN dim_products p ON f.product_id = p.product_id
    WHERE order_year BETWEEN 2015 AND 2025
    GROUP BY order_year, p.subcategory
""")
subcats = df_yoy["subcategory"].unique()
n_cols = 3
n_rows_plot = -(-len(subcats) // n_cols)  # ceiling division
fig, axes = plt.subplots(n_rows_plot, n_cols, figsize=(15, n_rows_plot * 3), sharex=True)
axes_flat = axes.flatten()

for i, subcat in enumerate(subcats):
    ax = axes_flat[i]
    data = df_yoy[df_yoy["subcategory"] == subcat].sort_values("order_year")
    ax.plot(data["order_year"], data["revenue"] / 1e6, marker="o", color=PALETTE[i % len(PALETTE)], linewidth=2)
    ax.set_title(subcat, fontsize=10); ax.set_ylabel("₹M")
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle("YoY Revenue Growth by Subcategory", fontsize=13, fontweight="bold")
plt.tight_layout()
save_chart(fig, "20_yoy_subcategory_growth.png")

# %%
chart_files = sorted(CHARTS_DIR.glob("*.png"))
print(f"\n{'='*50}")
print(f"EDA COMPLETE: {len(chart_files)} charts saved to {CHARTS_DIR}")
print("="*50)
for f in chart_files:
    print(f"  ✓ {f.name}")
print("\nSection 2 complete. Run feature store setup next.")
