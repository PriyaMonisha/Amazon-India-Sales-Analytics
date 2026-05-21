# %% [markdown]
# # Notebook 2: EDA -- 20 Analyses from PostgreSQL
# All matplotlib/seaborn (fast, no browser). Charts to artifacts/charts/
# Run: APP_ENV=local python notebooks/02_eda.py

# %%
import logging, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from sqlalchemy import text
import config
from src.etl.load import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("eda")

CHARTS = config.ARTIFACTS_DIR / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)
engine = get_engine(config.DB_URL)
logger.info("Connected to PostgreSQL")

PAL = sns.color_palette("Set2")
sns.set_theme(style="whitegrid", palette="Set2")

def save(fig, name):
    fig.savefig(CHARTS / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", name)

def q(sql):
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


# %% Q1: Revenue Trend 2015-2025
df = q("SELECT order_year, SUM(final_amount_inr)/1e9 AS rev, COUNT(*) AS orders FROM fact_transactions WHERE order_year BETWEEN 2015 AND 2025 GROUP BY order_year ORDER BY order_year")
df["yoy"] = df["rev"].pct_change() * 100
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df["order_year"], df["rev"], marker="o", lw=2.5, color=PAL[0])
ax.fill_between(df["order_year"], df["rev"], alpha=0.15, color=PAL[0])
for _, r in df.iterrows():
    if not pd.isna(r["yoy"]):
        val = r["yoy"]
        # Fix: show sign correctly — no "+-" for negatives
        label = f"+{val:.0f}%" if val >= 0 else f"{val:.0f}%"
        color = "green" if val >= 0 else "red"
        ax.annotate(label, (r["order_year"], r["rev"]),
                    textcoords="offset points", xytext=(0,8), ha="center", fontsize=8, color=color)
ax.set_title("Amazon India Revenue Trend 2015-2025", fontweight="bold", fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Revenue (INR Billion)")
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); save(fig, "01_revenue_trend.png")


# %% Q2: Seasonal Heatmap
df = q("SELECT order_year, order_month, SUM(final_amount_inr)/1e6 AS rev FROM fact_transactions WHERE order_year BETWEEN 2015 AND 2025 GROUP BY order_year, order_month")
pivot = df.pivot(index="order_year", columns="order_month", values="rev").fillna(0)
pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
# Fix: taller figure + explicit y-axis tick spacing so years are readable
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.4,
            annot_kws={"size": 9})
ax.set_title("Monthly Revenue Heatmap (INR Million)", fontweight="bold", fontsize=13)
ax.set_ylabel("Year"); ax.set_xlabel("")
# Fix stacked y-axis: set ytick labels individually with proper spacing
ax.set_yticklabels([str(int(y)) for y in pivot.index], rotation=0, fontsize=9)
plt.tight_layout(); save(fig, "02_seasonal_heatmap.png")


# %% Q3: RFM Segments
df = q("SELECT customer_id, MAX(order_year*100+order_month) AS lp, COUNT(*) AS freq, SUM(final_amount_inr) AS mon FROM fact_transactions GROUP BY customer_id")
df["recency"] = df["lp"].max() - df["lp"]

def safe_qcut(s, q_bins, asc=True):
    labs = list(range(1,q_bins+1)) if asc else list(range(q_bins,0,-1))
    try:
        return pd.qcut(s, q=q_bins, labels=labs, duplicates="drop")
    except Exception:
        return pd.cut(s.rank(method="first"), bins=q_bins, labels=labs)

df["r_s"] = safe_qcut(df["recency"], 5, asc=False)
df["f_s"] = safe_qcut(df["freq"], 5)
df["m_s"] = safe_qcut(df["mon"], 5)

def seg(row):
    try:
        rv, fv, mv = int(str(row["r_s"])), int(str(row["f_s"])), int(str(row["m_s"]))
        if rv>=4 and fv>=4 and mv>=4: return "Champions"
        if rv>=3 and fv>=3 and mv>=3: return "Loyal"
        if rv>=4 and fv<=2:           return "Promising"
        if rv<=2 and fv>=3:           return "At Risk"
    except Exception:
        pass
    return "Hibernating"

df["segment"] = df.apply(seg, axis=1)
counts = df["segment"].value_counts()
fig, ax = plt.subplots(figsize=(7,4))
bars = ax.bar(counts.index, counts.values, color=PAL[:len(counts)])
ax.bar_label(bars); ax.set_title("RFM Customer Segments", fontweight="bold")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout(); save(fig, "03_rfm_segments.png")


# %% Q4: Payment Method Evolution
df = q("SELECT order_year, payment_method, COUNT(*) AS n FROM fact_transactions WHERE order_year BETWEEN 2015 AND 2025 GROUP BY order_year, payment_method")
pivot = df.pivot(index="order_year", columns="payment_method", values="n").fillna(0)
pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(12, 5))
pct.plot.area(ax=ax, stacked=True, alpha=0.85, colormap="Set2")
ax.set_title("Payment Method Market Share 2015-2025", fontweight="bold", fontsize=13)
ax.set_ylabel("Share (%)"); ax.set_xlabel("Year")   # Fix: "order_year" -> "Year"
ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9, title="Payment Method")
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); save(fig, "04_payment_evolution.png")


# %% Q5: Subcategory Performance
df = q("SELECT p.subcategory, SUM(f.final_amount_inr)/1e9 AS rev FROM fact_transactions f JOIN dim_products p ON f.product_id=p.product_id GROUP BY p.subcategory ORDER BY rev DESC")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cols = PAL[:len(df)]

# Bar chart — show actual value with smart formatting (not "0.0 B" for small ones)
def smart_label(v):
    if v >= 0.1:   return f"{v:.1f} B"
    return f"{v*1000:.0f} M"

bars = axes[0].barh(df["subcategory"][::-1], df["rev"][::-1], color=cols[::-1])
axes[0].bar_label(bars, labels=[smart_label(v) for v in df["rev"][::-1]], fontsize=9, padding=3)
axes[0].set_title("Revenue by Subcategory (INR Bn)", fontweight="bold")
axes[0].set_xlabel("INR Billion")

# Pie chart — fix label overlap: use legend for small slices, pctdistance adjustment
threshold = 3.0   # slices below 3% get no label to avoid clutter
labels_pie = [s if (v / df["rev"].sum() * 100) >= threshold else "" for s, v in zip(df["subcategory"], df["rev"])]
wedges, texts, autotexts = axes[1].pie(
    df["rev"], labels=labels_pie, autopct=lambda p: f"{p:.1f}%" if p >= threshold else "",
    colors=cols, startangle=90, pctdistance=0.78, labeldistance=1.08
)
for t in autotexts:
    t.set_fontsize(8)
# Legend covers all slices including small ones
axes[1].legend(wedges, df["subcategory"], loc="lower center",
               bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
axes[1].set_title("Market Share", fontweight="bold")
plt.suptitle("Subcategory Performance", fontweight="bold", fontsize=13)
plt.tight_layout(); save(fig, "05_subcategory_performance.png")

df2 = q("SELECT f.order_year, p.subcategory, SUM(f.final_amount_inr)/1e6 AS rev FROM fact_transactions f JOIN dim_products p ON f.product_id=p.product_id WHERE f.order_year BETWEEN 2015 AND 2025 GROUP BY f.order_year, p.subcategory")
fig, ax = plt.subplots(figsize=(11,4))
for i, sub in enumerate(df2["subcategory"].unique()):
    d = df2[df2["subcategory"]==sub].sort_values("order_year")
    ax.plot(d["order_year"], d["rev"], marker="o", label=sub, color=PAL[i%len(PAL)], lw=2)
ax.set_title("Subcategory Revenue Growth (INR Million)", fontweight="bold")
ax.legend(fontsize=8); ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout(); save(fig, "05b_subcategory_growth.png")


# %% Q6: Prime Membership Impact
df = q("SELECT c.is_prime_member, AVG(f.final_amount_inr) AS aov, COUNT(*) AS orders, COUNT(DISTINCT f.customer_id) AS custs FROM fact_transactions f JOIN dim_customers c ON f.customer_id=c.customer_id GROUP BY c.is_prime_member")
df["label"] = df["is_prime_member"].map({True:"Prime", False:"Non-Prime"}).fillna("Unknown")
df["freq"] = df["orders"] / df["custs"]
fig, axes = plt.subplots(1,3,figsize=(13,4))
for ax, col, title, fmt in zip(axes, ["aov","freq","orders"], ["Avg Order Value (INR)","Orders/Customer","Total Orders"], ["{:,.0f}","{:.1f}","{:,.0f}"]):
    bars = ax.bar(df["label"], df[col], color=PAL[:len(df)])
    ax.bar_label(bars, labels=[fmt.format(v) for v in df[col]], fontsize=9)
    ax.set_title(title)
plt.suptitle("Prime vs Non-Prime", fontweight="bold"); plt.tight_layout()
save(fig, "06_prime_impact.png")


# %% Q7: Geographic & Tier
df_s = q("SELECT c.customer_state, SUM(f.final_amount_inr)/1e9 AS rev FROM fact_transactions f JOIN dim_customers c ON f.customer_id=c.customer_id GROUP BY c.customer_state ORDER BY rev DESC LIMIT 15")
df_t = q("SELECT c.customer_tier, SUM(f.final_amount_inr)/1e9 AS rev FROM fact_transactions f JOIN dim_customers c ON f.customer_id=c.customer_id GROUP BY c.customer_tier ORDER BY rev DESC")
fig, axes = plt.subplots(1,2,figsize=(14,5))
bars = axes[0].barh(df_s["customer_state"][::-1], df_s["rev"][::-1], color=PAL[:len(df_s)])
axes[0].bar_label(bars, fmt="%.1f B", fontsize=8); axes[0].set_title("Top States (INR Bn)")
bars2 = axes[1].bar(df_t["customer_tier"], df_t["rev"], color=PAL[:len(df_t)])
axes[1].bar_label(bars2, fmt="%.1f B"); axes[1].set_title("Revenue by Tier (INR Bn)")
plt.suptitle("Geographic & Tier", fontweight="bold"); plt.tight_layout()
save(fig, "07_geographic_tier.png")


# %% Q8: Festival Impact
df = q("SELECT is_festival_sale, festival_name, final_amount_inr FROM fact_transactions")
grp = df.groupby("is_festival_sale")["final_amount_inr"].agg(["mean","count","sum"]).reset_index()
grp["label"] = grp["is_festival_sale"].map({True:"Festival Sale", False:"Regular Sale"})
grp = grp.sort_values("is_festival_sale")  # Regular first, Festival second (consistent left-to-right)

fest_rev = (df[df["is_festival_sale"]==True]
            .groupby("festival_name")["final_amount_inr"].sum()
            .sort_values(ascending=True) / 1e6)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fix: use consistent colors (green=Regular/normal, orange=Festival/highlight)
bar_colors = [PAL[0], PAL[1]]
bars = axes[0].bar(grp["label"], grp["mean"] / 1000, color=bar_colors, width=0.5, edgecolor="white", lw=1.5)
for bar, (_, row) in zip(bars, grp.iterrows()):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"INR {row['mean']/1000:.1f}K\n(n={int(row['count']):,})",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].set_title("Avg Order Value: Festival vs Regular Sale", fontweight="bold")
axes[0].set_ylabel("Avg Order Value (INR K)")
axes[0].set_ylim(0, grp["mean"].max() / 1000 * 1.3)

if len(fest_rev):
    colors_f = [PAL[i % len(PAL)] for i in range(len(fest_rev))]
    bars2 = axes[1].barh(fest_rev.index, fest_rev.values, color=colors_f, edgecolor="white")
    axes[1].bar_label(bars2, fmt="%.0f M", fontsize=8, padding=3)
    axes[1].set_title("Total Revenue by Festival (INR Million)", fontweight="bold")
    axes[1].set_xlabel("INR Million")

plt.suptitle("Festival Sales Analysis", fontweight="bold", fontsize=13)
plt.tight_layout(); save(fig, "08_festival_impact.png")


# %% Q9: Customer Tier Behaviour
df = q("SELECT c.customer_tier, f.final_amount_inr, f.payment_method FROM fact_transactions f JOIN dim_customers c ON f.customer_id=c.customer_id")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
order_tiers = df.groupby("customer_tier")["final_amount_inr"].median().sort_values(ascending=False).index

sns.boxplot(data=df, x="customer_tier", y="final_amount_inr", order=order_tiers,
            palette="Set2", ax=axes[0], showfliers=False)  # hide outliers to fix 1e6 axis
axes[0].set_title("Order Value Distribution by Tier\n(outliers hidden for clarity)", fontweight="bold")
axes[0].set_ylabel("Order Value (INR)")  # Fix: remove confusing "1e6" by hiding extreme outliers
axes[0].set_xlabel("Customer Tier")
# Format y-axis as K (thousands)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

pp = df.groupby(["customer_tier","payment_method"]).size().unstack(fill_value=0)
pp.div(pp.sum(axis=1), axis=0).mul(100).plot.bar(ax=axes[1], stacked=True, colormap="Set2", alpha=0.9)
axes[1].set_title("Payment Method Mix by Tier (%)", fontweight="bold")
axes[1].set_ylabel("Share (%)"); axes[1].set_xlabel("Customer Tier")
axes[1].legend(fontsize=7, bbox_to_anchor=(1, 1), title="Payment")
axes[1].tick_params(axis="x", rotation=20)
plt.suptitle("Customer Tier Behaviour", fontweight="bold", fontsize=13)
plt.tight_layout(); save(fig, "09_tier_behaviour.png")


# %% Q10: Age Group Preferences
df = q("SELECT c.customer_age_group, p.subcategory, SUM(f.final_amount_inr)/1e6 AS rev FROM fact_transactions f JOIN dim_customers c ON f.customer_id=c.customer_id JOIN dim_products p ON f.product_id=p.product_id WHERE c.customer_age_group IS NOT NULL AND c.customer_age_group != 'nan' GROUP BY c.customer_age_group, p.subcategory")
pivot = df.pivot(index="customer_age_group", columns="subcategory", values="rev").fillna(0)
# Fix: sort rows by age group in logical order, not alphabetical
age_order = ["18-25", "26-35", "36-45", "46-55", "55+"]
pivot = pivot.reindex([a for a in age_order if a in pivot.index])
# Fix: taller figure so rows are readable — no more stacked labels
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", ax=ax,
            linewidths=0.4, annot_kws={"size": 10})
ax.set_title("Revenue (INR Million) by Age Group x Subcategory", fontweight="bold", fontsize=13)
ax.set_xlabel("Product Subcategory"); ax.set_ylabel("Customer Age Group")
# Fix: y-axis labels horizontal and readable
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); save(fig, "10_age_group.png")


# %% Q11: Price vs Demand
df = q("SELECT f.discounted_price_inr, f.quantity, f.discount_percent, p.subcategory FROM fact_transactions f JOIN dim_products p ON f.product_id=p.product_id WHERE f.discounted_price_inr IS NOT NULL LIMIT 10000")
fig, axes = plt.subplots(1,2,figsize=(13,4))
for i, sub in enumerate(df["subcategory"].unique()[:6]):
    d = df[df["subcategory"]==sub]
    axes[0].scatter(d["discounted_price_inr"], d["quantity"], alpha=0.4, label=sub, color=PAL[i%len(PAL)], s=15)
axes[0].set_title("Price vs Demand"); axes[0].set_xlabel("Price (INR)"); axes[0].set_ylabel("Qty"); axes[0].legend(fontsize=7)
num_c = ["discounted_price_inr","discount_percent","quantity"]
sns.heatmap(df[num_c].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
axes[1].set_title("Price Correlation Matrix")
plt.suptitle("Price Sensitivity", fontweight="bold"); plt.tight_layout()
save(fig, "11_price_vs_demand.png")


# %% Q12: Brand Performance
df = q("SELECT p.brand, SUM(f.final_amount_inr)/1e9 AS rev, AVG(f.discount_percent) AS disc, AVG(f.product_rating) AS rating FROM fact_transactions f JOIN dim_products p ON f.product_id=p.product_id GROUP BY p.brand ORDER BY rev DESC LIMIT 15")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Fix: smart labels — show "M" for sub-billion brands instead of "0.0 B"
bar_labels = [smart_label(v) for v in df["rev"][::-1]]
bars = axes[0].barh(df["brand"][::-1], df["rev"][::-1],
                    color=[PAL[i % len(PAL)] for i in range(len(df))])
axes[0].bar_label(bars, labels=bar_labels, fontsize=8, padding=3)
axes[0].set_title("Top 15 Brands by Revenue", fontweight="bold")
axes[0].set_xlabel("Revenue (INR Billion)")

# Bubble chart: fix overlapping brand labels with adjustText-style offsets
scatter_sizes = (df["rev"] / df["rev"].max() * 800 + 50).clip(lower=50)
sc = axes[1].scatter(df["disc"], df["rating"], s=scatter_sizes,
                     c=range(len(df)), cmap="Set2", alpha=0.85, edgecolors="white", lw=0.5)
# Annotate with slight vertical offset to avoid overlap with dot
for _, row in df.iterrows():
    offset = 0.015 if row["rev"] > 0.1 else -0.025
    axes[1].annotate(row["brand"], (row["disc"], row["rating"] + offset),
                     fontsize=7, ha="center", va="bottom")
axes[1].set_title("Brand: Avg Discount vs Avg Rating\n(bubble size = revenue)", fontweight="bold")
axes[1].set_xlabel("Avg Discount (%)"); axes[1].set_ylabel("Avg Product Rating")
plt.suptitle("Brand Performance", fontweight="bold", fontsize=13)
plt.tight_layout(); save(fig, "12_brand_performance.png")


# %% Q13: Return Rate
df = q("SELECT p.subcategory, f.payment_method, f.return_status, COUNT(*) AS n FROM fact_transactions f JOIN dim_products p ON f.product_id=p.product_id GROUP BY p.subcategory, f.payment_method, f.return_status")
tot = df.groupby("subcategory")["n"].sum()
ret = df[df["return_status"]=="Returned"].groupby("subcategory")["n"].sum()
rr = (ret/tot*100).fillna(0).sort_values(ascending=False).reset_index()
rr.columns = ["subcategory","rate"]
pay_tot = df.groupby("payment_method")["n"].sum()
pay_ret = df[df["return_status"]=="Returned"].groupby("payment_method")["n"].sum()
pay_rr = (pay_ret/pay_tot*100).fillna(0).sort_values(ascending=False)
fig, axes = plt.subplots(1,2,figsize=(13,4))
bars = axes[0].bar(rr["subcategory"], rr["rate"], color=PAL[:len(rr)])
axes[0].bar_label(bars, fmt="%.1f%%"); axes[0].set_title("Return Rate by Subcategory"); axes[0].tick_params(axis="x", rotation=30)
bars2 = axes[1].bar(pay_rr.index, pay_rr.values, color=PAL[:len(pay_rr)])
axes[1].bar_label(bars2, fmt="%.1f%%"); axes[1].set_title("Return Rate by Payment Method"); axes[1].tick_params(axis="x", rotation=30)
plt.suptitle("Return Rate Analysis", fontweight="bold"); plt.tight_layout()
save(fig, "13_return_rate.png")


# %% Q14: Cohort Retention
df = q("SELECT customer_id, order_year FROM fact_transactions WHERE order_year BETWEEN 2015 AND 2025")
first = df.groupby("customer_id")["order_year"].min().reset_index()
first.columns = ["customer_id", "cohort"]
df = df.merge(first, on="customer_id")
cohort_m = df.groupby(["cohort", "order_year"])["customer_id"].nunique().unstack(fill_value=0)

# Fix: safe division — avoid NaN from cohorts with 0 first-year customers
cohort_sizes = cohort_m.iloc[:, 0].replace(0, np.nan)
pct_m = cohort_m.divide(cohort_sizes, axis=0) * 100
pct_m = pct_m.fillna(0)

# Only plot cohorts that have data (cohort size > 5 to be meaningful)
valid = cohort_sizes[cohort_sizes >= 5].index
pct_m = pct_m.loc[pct_m.index.isin(valid)] if len(valid) > 0 else pct_m

fig, ax = plt.subplots(figsize=(14, max(5, len(pct_m) * 0.6)))
# Fix colorbar: explicitly set vmin=0, vmax=100 so it shows 0-100% range
sns.heatmap(pct_m, annot=True, fmt=".0f", cmap="Blues", ax=ax,
            linewidths=0.4, vmin=0, vmax=100,
            annot_kws={"size": 9})
ax.set_title("Customer Cohort Retention\n(% of cohort still purchasing in each year)",
             fontweight="bold", fontsize=12)
ax.set_xlabel("Order Year"); ax.set_ylabel("First Purchase Year (Cohort)")
# Fix: y-axis labels horizontal
ax.set_yticklabels([str(int(y)) for y in pct_m.index], rotation=0, fontsize=9)
plt.tight_layout(); save(fig, "14_cohort_retention.png")


# %% Q15: CLV Distribution
df = q("SELECT customer_id, SUM(final_amount_inr) AS clv FROM fact_transactions GROUP BY customer_id")
cap = df["clv"].quantile(0.99)
fig, ax = plt.subplots(figsize=(10,4))
ax.hist(df["clv"].clip(upper=cap)/1000, bins=50, color=PAL[0], edgecolor="white", alpha=0.85)
ax.axvline(df["clv"].mean()/1000, color="red", ls="--", lw=1.5, label=f"Mean INR {df['clv'].mean()/1000:.1f}K")
ax.axvline(df["clv"].median()/1000, color="green", ls=":", lw=1.5, label=f"Median INR {df['clv'].median()/1000:.1f}K")
ax.set_title("Customer Lifetime Value Distribution", fontweight="bold")
ax.set_xlabel("CLV (INR K)"); ax.set_ylabel("Customers"); ax.legend()
plt.tight_layout(); save(fig, "15_clv_distribution.png")


# %% Q16: Delivery Performance
df = q("SELECT f.delivery_type, f.delivery_days, c.customer_tier FROM fact_transactions f JOIN dim_customers c ON f.customer_id=c.customer_id WHERE f.delivery_days IS NOT NULL AND f.delivery_days >= 0")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Fix: remove the "delivery_type" column-name showing as x-axis label
sns.boxplot(data=df, x="delivery_type", y="delivery_days",
            palette="Set2", ax=axes[0], showfliers=False,
            order=["Same Day", "Express", "Standard"])
axes[0].set_title("Delivery Days by Delivery Type\n(outliers hidden)", fontweight="bold")
axes[0].set_ylabel("Days"); axes[0].set_xlabel("")   # Fix: remove "delivery_type" axis label

# Add median annotation per box
for i, dtype in enumerate(["Same Day", "Express", "Standard"]):
    med = df[df["delivery_type"] == dtype]["delivery_days"].median()
    axes[0].text(i, med + 0.1, f"med {med:.1f}d", ha="center", fontsize=8, color="darkblue")

avg = df.groupby("customer_tier")["delivery_days"].mean().sort_values()
colors_t = [PAL[i % len(PAL)] for i in range(len(avg))]
bars = axes[1].bar(avg.index, avg.values, color=colors_t, edgecolor="white")
axes[1].bar_label(bars, fmt="%.1f d", fontsize=9)
axes[1].set_title("Avg Delivery Days by Customer Tier", fontweight="bold")
axes[1].set_xlabel("Customer Tier"); axes[1].set_ylabel("Days")
plt.suptitle("Delivery Performance", fontweight="bold", fontsize=13)
plt.tight_layout(); save(fig, "16_delivery_performance.png")


# %% Q17: Discount Effectiveness
df = q("SELECT discount_percent, quantity, final_amount_inr FROM fact_transactions WHERE discount_percent IS NOT NULL LIMIT 20000")
bins_d = [0,10,20,30,40,50,70]; labels_d = ["0-10","10-20","20-30","30-40","40-50","50+"]
df["disc_bin"] = pd.cut(df["discount_percent"], bins=bins_d, labels=labels_d)
qty_b = df.groupby("disc_bin", observed=True)["quantity"].mean()
rev_b = df.groupby("disc_bin", observed=True)["final_amount_inr"].sum()/1e6
fig, axes = plt.subplots(1,2,figsize=(13,4))
bars = axes[0].bar(qty_b.index, qty_b.values, color=PAL[:len(qty_b)])
axes[0].bar_label(bars, fmt="%.2f"); axes[0].set_title("Avg Qty by Discount Range"); axes[0].set_xlabel("Discount %")
bars2 = axes[1].bar(rev_b.index, rev_b.values, color=PAL[:len(rev_b)])
axes[1].bar_label(bars2, fmt="%.0f M"); axes[1].set_title("Revenue by Discount Range (INR M)")
plt.suptitle("Discount Effectiveness", fontweight="bold"); plt.tight_layout()
save(fig, "17_discount_effectiveness.png")


# %% Q18: Product Rating
df = q("SELECT p.subcategory, f.product_rating FROM fact_transactions f JOIN dim_products p ON f.product_id=p.product_id WHERE f.product_rating IS NOT NULL")
fig, ax = plt.subplots(figsize=(11,4))
sns.boxplot(data=df, x="subcategory", y="product_rating", palette="Set2", ax=ax)
ax.set_title("Product Rating by Subcategory", fontweight="bold"); ax.set_ylabel("Rating (out of 5)")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); save(fig, "18_product_rating.png")


# %% Q19: Pareto Revenue Concentration
df = q("SELECT product_id, SUM(final_amount_inr) AS rev FROM fact_transactions GROUP BY product_id ORDER BY rev DESC")
df["cum"] = df["rev"].cumsum() / df["rev"].sum() * 100
df["pp"] = np.arange(1,len(df)+1) / len(df) * 100
top80 = df[df["cum"]<=80]["pp"].max() if (df["cum"]<=80).any() else 100
fig, ax = plt.subplots(figsize=(9,4))
ax.plot(df["pp"], df["cum"], color=PAL[0], lw=2)
ax.axhline(80, color="red", ls="--", alpha=0.7, label="80% revenue")
ax.axvline(top80, color="orange", ls="--", alpha=0.7, label=f"Top {top80:.0f}% products")
ax.set_title("Pareto: Revenue Concentration", fontweight="bold")
ax.set_xlabel("% Products"); ax.set_ylabel("Cumulative % Revenue"); ax.legend()
plt.tight_layout(); save(fig, "19_revenue_pareto.png")
print(f"Top {top80:.0f}% of products generate 80% of revenue")


# %% Q20: YoY Subcategory Growth (small multiples)
df = q("SELECT f.order_year, p.subcategory, SUM(f.final_amount_inr)/1e6 AS rev FROM fact_transactions f JOIN dim_products p ON f.product_id=p.product_id WHERE f.order_year BETWEEN 2015 AND 2025 GROUP BY f.order_year, p.subcategory")
subcats = sorted(df["subcategory"].unique())
nc = 3; nr = -(-len(subcats) // nc)
fig, axes = plt.subplots(nr, nc, figsize=(15, nr * 3.5), sharex=True)
axf = axes.flatten() if nr > 1 else list(axes)
for i, sub in enumerate(subcats):
    d = df[df["subcategory"] == sub].sort_values("order_year")
    axf[i].plot(d["order_year"], d["rev"], marker="o", color=PAL[i % len(PAL)], lw=2.5)
    axf[i].fill_between(d["order_year"], d["rev"], alpha=0.1, color=PAL[i % len(PAL)])
    axf[i].set_title(sub, fontsize=11, fontweight="bold")
    axf[i].set_ylabel("INR Million")
    # Fix: show every year on x-axis (not just even years)
    axf[i].xaxis.set_major_locator(mticker.MultipleLocator(2))
    axf[i].tick_params(axis="x", rotation=45, labelsize=8)
    # Add peak annotation
    peak_idx = d["rev"].idxmax()
    if not pd.isna(peak_idx):
        peak_row = d.loc[peak_idx]
        axf[i].annotate(f"Peak\n{peak_row['order_year']:.0f}",
                        xy=(peak_row["order_year"], peak_row["rev"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color="darkblue")
for j in range(len(subcats), len(axf)):
    axf[j].set_visible(False)
plt.suptitle("YoY Revenue Growth by Subcategory (INR Million)", fontweight="bold", fontsize=13, y=1.01)
plt.tight_layout(); save(fig, "20_yoy_subcategory.png")


# %% Summary
charts = sorted(CHARTS.glob("*.png"))
print(f"\nEDA COMPLETE: {len(charts)} charts saved")
for c in charts:
    print(f"  {c.name}")
print("\nSection 2 done. Next: Section 3 (Feature Store).")
