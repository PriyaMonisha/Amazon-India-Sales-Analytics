# %% [markdown]
# # Notebook 1: Data Cleaning -- Amazon India Sales Analytics
#
# **What this notebook does:**
# Loads all 11 raw CSVs from data/raw/, identifies and fixes 10 real data quality
# issues found in the dataset, and saves a cleaned merged CSV to data/processed/.
#
# **Dataset:** 1,127,609 transactions | 2015-2025 | 34 columns + product catalog
#
# **This is the analysis/portfolio notebook.**
# The production ETL runner (that loads to PostgreSQL) is in 01_data_engineering.py
#
# Run: `python notebooks/01_data_cleaning.py`

# %%
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent
    if not (PROJECT_ROOT / "config.py").exists():
        PROJECT_ROOT = Path.cwd()

sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root : {PROJECT_ROOT}")
print(f"Raw data dir : {RAW_DIR}")

# %% [markdown]
# ## Step 1: Load All 11 Raw CSVs

# %%
csv_files = sorted(RAW_DIR.glob("amazon_india_20*.csv"))
print(f"Found {len(csv_files)} yearly CSV files:")
for f in csv_files:
    print(f"  {f.name}")

# %%
dfs = []
for f in csv_files:
    df = pd.read_csv(f, low_memory=False)
    df["source_year"] = int(f.stem.split("_")[-1])
    dfs.append(df)
    print(f"  {f.name}: {len(df):,} rows x {len(df.columns)} cols")

df_raw = pd.concat(dfs, ignore_index=True)
print(f"\nMerged: {len(df_raw):,} rows x {len(df_raw.columns)} columns")

# %%
# Load product catalog separately
df_products_raw = pd.read_csv(RAW_DIR / "amazon_india_products_catalog.csv")
print(f"Product catalog: {len(df_products_raw):,} rows x {len(df_products_raw.columns)} columns")
print(df_products_raw.head(3).to_string())

# %%
# Initial overview
print("\n=== Column Overview ===")
overview = pd.DataFrame({
    "dtype": df_raw.dtypes,
    "nulls": df_raw.isna().sum(),
    "null_%": (df_raw.isna().sum() / len(df_raw) * 100).round(2),
    "unique": df_raw.nunique(),
}).sort_values("null_%", ascending=False)
print(overview.to_string())

# %% [markdown]
# ## Challenge 1 -- Price Column Has Commas (Object dtype)
#
# **Problem:** `original_price_inr` is stored as object (string).
# 101,569 rows have Indian comma formatting: "21,947.26" instead of 21947.26
# A direct `astype(float)` would fail on these rows.

# %%
# BEFORE: Show the problem
price_samples = df_raw["original_price_inr"].head(10).tolist()
comma_count = df_raw["original_price_inr"].astype(str).str.contains(",").sum()
print(f"BEFORE -- original_price_inr dtype: {df_raw['original_price_inr'].dtype}")
print(f"Rows with commas in price: {comma_count:,}")
print(f"Sample values: {price_samples}")

# %%
# FIX: Strip rupee symbol (both INR  and Rs), commas, spaces -> float
import re as _re

def clean_price(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Remove currency symbols and prefixes: rupee symbol, Rs, Rs., INR
    s = _re.sub(r"^(Rs\.?\s*|INR\s*)", "", s, flags=_re.IGNORECASE)
    s = s.replace("₹", "")  # rupee unicode symbol
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None

df = df_raw.copy()
df["original_price_inr"] = df["original_price_inr"].apply(clean_price)

# Impute remaining nulls with subcategory median
nulls_after_parse = df["original_price_inr"].isna().sum()
if nulls_after_parse > 0:
    subcat_median = df.groupby("subcategory")["original_price_inr"].transform("median")
    df["original_price_inr"] = df["original_price_inr"].fillna(subcat_median)

# AFTER: Verify (avoid rupee symbol for Windows cp1252 console)
print(f"\nAFTER -- original_price_inr dtype: {df['original_price_inr'].dtype}")
print(f"Nulls after parse (pre-imputation): {nulls_after_parse:,} -> now {df['original_price_inr'].isna().sum():,}")
print(f"Min: INR {df['original_price_inr'].min():,.2f} | Max: INR {df['original_price_inr'].max():,.2f}")
print(f"Mean: INR {df['original_price_inr'].mean():,.2f}")

# %% [markdown]
# ## Challenge 2 -- Delivery Days Has Mixed Text + Numbers (Object dtype)
#
# **Problem:** `delivery_days` is a string column with 13 distinct values:
# '-1' (invalid), '1-2 days' (range), 'Express' (text), 'Same Day' (text),
# plus regular numbers '0'-'15'.

# %%
# BEFORE
print(f"BEFORE -- delivery_days dtype: {df_raw['delivery_days'].dtype}")
print(f"All unique values: {sorted(df_raw['delivery_days'].unique())}")

# %%
# FIX: Parse text -> numeric, cap outliers, negative -> NaN
def clean_delivery_days(val):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ("same day", "same-day"):
        return 0.0
    if s == "express":
        return 1.0          # Express = next day
    import re
    m = re.match(r"(\d+)\s*[--]\s*(\d+)", s)
    if m:                   # "1-2 days" -> 1.5
        return (float(m.group(1)) + float(m.group(2))) / 2
    try:
        v = float(re.sub(r"[^\d.]", "", s))
        return v if 0 <= v <= 30 else None   # Cap at 30; negative -> None
    except ValueError:
        return None

df["delivery_days"] = df["delivery_days"].apply(clean_delivery_days)

# AFTER
print(f"\nAFTER -- delivery_days dtype: {df['delivery_days'].dtype}")
print(f"Remaining nulls (invalid values): {df['delivery_days'].isna().sum():,}")
print(f"Min: {df['delivery_days'].min()} | Max: {df['delivery_days'].max()}")
print(f"Value distribution:\n{df['delivery_days'].value_counts().sort_index().head(10)}")

# %% [markdown]
# ## Challenge 3 -- Boolean Columns Have 8 Different Representations
#
# **Problem:** `is_prime_member`, `is_festival_sale`, `is_prime_eligible` all have
# 8 variants: '0', '1', 'FALSE', 'False', 'No', 'TRUE', 'True', 'Yes'
# Some years used True/False, others used Yes/No, others used 0/1.

# %%
# BEFORE
for col in ["is_prime_member", "is_festival_sale", "is_prime_eligible"]:
    print(f"BEFORE -- {col} unique values: {sorted(df_raw[col].unique())}")

# %%
# FIX: Unified bool parser
def to_bool(val):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    return None

for col in ["is_prime_member", "is_festival_sale", "is_prime_eligible"]:
    df[col] = df[col].apply(to_bool)

# AFTER
print("\nAFTER:")
for col in ["is_prime_member", "is_festival_sale", "is_prime_eligible"]:
    print(f"  {col}: {df[col].unique()} | True={df[col].sum():,} | False={df[col].eq(False).sum():,} | Null={df[col].isna().sum():,}")

# %% [markdown]
# ## Challenge 4 -- Customer Rating Has 21 Format Variants
#
# **Problem:** `customer_rating` has 21 distinct string formats:
# - Plain number: '3', '4', '5'
# - Decimal: '3.5', '4.0'
# - Stars: '3.0 stars', '4.5 stars'
# - Fraction: '3/5', '4/5', '4.0/5.0'
# - Plus 341,696 nulls (30% missing -- expected, some customers don't rate)

# %%
# BEFORE
print(f"BEFORE -- customer_rating dtype: {df_raw['customer_rating'].dtype}")
print(f"All 21 variants: {sorted(df_raw['customer_rating'].dropna().unique())}")
print(f"Null count: {df_raw['customer_rating'].isna().sum():,} ({df_raw['customer_rating'].isna().mean()*100:.1f}%)")

# %%
# FIX: Parse all 21 variants -> float on 0-5 scale
import re as _re

def parse_rating(val):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    # "4 stars" or "4.5 stars"
    m = _re.match(r"^(\d+(?:\.\d+)?)\s*stars?$", s)
    if m:
        return float(m.group(1))
    # "3/5" or "4.0/5.0" -> normalise to /5 scale
    m = _re.match(r"^(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$", s)
    if m:
        num, denom = float(m.group(1)), float(m.group(2))
        if denom > 0:
            return round(num * 5.0 / denom, 2)
    try:
        return float(s)
    except ValueError:
        return None

df["customer_rating"] = df["customer_rating"].apply(parse_rating)

# Impute missing with product median (same product, similar quality)
product_medians = df.groupby("product_id")["customer_rating"].transform("median")
global_median = df["customer_rating"].median()
df["customer_rating"] = df["customer_rating"].fillna(product_medians).fillna(global_median)

# Flag whether rating was originally missing (useful feature for analysis)
df["rating_missing"] = df_raw["customer_rating"].isna().astype(int)

# AFTER
print(f"\nAFTER -- customer_rating dtype: {df['customer_rating'].dtype}")
print(f"Remaining nulls: {df['customer_rating'].isna().sum():,}")
print(f"Min: {df['customer_rating'].min():.1f} | Max: {df['customer_rating'].max():.1f} | Mean: {df['customer_rating'].mean():.2f}")
print(f"rating_missing flag: {df['rating_missing'].sum():,} rows had missing ratings")

# %% [markdown]
# ## Challenge 5 -- Order Date Has Two Format Standards
#
# **Problem:** `order_date` has two date formats mixed across years:
# - Standard ISO: '2015-01-25' (YYYY-MM-DD)
# - US format: '01/17/2015' (MM/DD/YYYY)
# - Also: '13/01/2015' (DD/MM/YYYY -- ambiguous day>12)

# %%
# BEFORE
print(f"BEFORE -- order_date dtype: {df_raw['order_date'].dtype}")
slash_ex = [d for d in df_raw["order_date"].unique() if "/" in str(d)][:5]
dash_ex = [d for d in df_raw["order_date"].unique() if "-" in str(d)][:5]
print(f"Slash format examples: {slash_ex}")
print(f"Dash format examples:  {dash_ex}")

# %%
# FIX: Try multiple date formats
from datetime import datetime

def parse_date(val):
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            pass
    return pd.NaT

df["order_date"] = df["order_date"].apply(parse_date)

# Flag invalid dates before replacing with sentinel
nat_count = df["order_date"].isna().sum()
print(f"Invalid/unparseable dates: {nat_count:,}")

# AFTER
print(f"\nAFTER -- order_date dtype: {df['order_date'].dtype}")
print(f"Date range: {df['order_date'].min().date()} -> {df['order_date'].max().date()}")
print(f"NaT (invalid) count: {df['order_date'].isna().sum():,}")

# %% [markdown]
# ## Challenge 6 -- Category Has 5 Spelling Variants (Should Be 1)
#
# **Problem:** `category` should always be "Electronics" but appears as:
# 'Electronics', 'ELECTRONICS', 'Electronic', 'Electronicss', 'Electronics & Accessories'

# %%
# BEFORE
print(f"BEFORE -- category unique values: {df_raw['category'].unique()}")
print(f"Value counts:")
print(df_raw["category"].value_counts().to_string())

# %%
# FIX: Lowercase -> strip -> map to standard name
CATEGORY_MAP = {
    "electronics": "Electronics",
    "electronic": "Electronics",
    "electronicss": "Electronics",
    "electronics & accessories": "Electronics",
}
df["category"] = (
    df["category"]
    .str.strip()
    .str.lower()
    .map(lambda v: CATEGORY_MAP.get(v, v.title()))
)

# AFTER
print(f"\nAFTER -- category unique values: {df['category'].unique()}")
print(f"Single value: {df['category'].nunique() == 1}")
# Since category is now all 'Electronics' -> can be dropped (no discriminatory power)
# We'll keep it but flag it
print("Note: category='Electronics' for all rows -> zero discriminatory value. Will drop before ML training.")

# %% [markdown]
# ## Challenge 7 -- Customer Age Group Has 135K Nulls

# %%
# BEFORE
print(f"BEFORE -- customer_age_group nulls: {df_raw['customer_age_group'].isna().sum():,}")
print(f"Value distribution:\n{df_raw['customer_age_group'].value_counts(dropna=False).to_string()}")

# %%
# FIX: Fill with mode within customer_tier group (same tier = similar demographics)
tier_mode = df.groupby("customer_tier")["customer_age_group"].transform(
    lambda x: x.mode()[0] if not x.mode().empty else "26-35"
)
df["customer_age_group"] = df["customer_age_group"].fillna(tier_mode)

# AFTER
print(f"\nAFTER -- customer_age_group nulls: {df['customer_age_group'].isna().sum():,}")
print(f"Distribution:\n{df['customer_age_group'].value_counts().to_string()}")

# %% [markdown]
# ## Challenge 8 -- Duplicate Rows (_DUP suffix + exact duplicates)
#
# **Problem:** Two types of duplicates:
# 1. transaction_ids ending with `_DUP` (explicitly flagged by source system)
# 2. Exact row duplicates from pipeline re-runs

# %%
# BEFORE
dup_flag = df["transaction_id"].str.endswith("_DUP", na=False)
exact_dups = df.duplicated(keep="first")
print(f"BEFORE -- rows with _DUP suffix: {dup_flag.sum():,}")
print(f"Exact duplicate rows: {exact_dups.sum():,}")
print(f"Total rows: {len(df):,}")

# %%
# FIX: Remove both types
remove_mask = dup_flag | exact_dups
df = df[~remove_mask].reset_index(drop=True)

# AFTER
print(f"\nAFTER -- Rows removed: {remove_mask.sum():,}")
print(f"Clean rows remaining: {len(df):,}")
dup_check = df["transaction_id"].duplicated().sum()
print(f"Remaining duplicate transaction_ids: {dup_check}")

# %% [markdown]
# ## Challenge 9 -- Delivery Charges Column Is Useless
#
# **Problem:** `delivery_charges` has 90,201 nulls and the remaining values are all 0.
# This column carries no signal -- Amazon India uses free delivery on most orders.

# %%
print(f"BEFORE -- delivery_charges unique values: {df_raw['delivery_charges'].unique()}")
print(f"delivery_charges nulls: {df_raw['delivery_charges'].isna().sum():,}")
print(f"Non-zero values: {(df_raw['delivery_charges'] != 0).sum():,}")

# %%
# FIX: Drop the column -- zero-variance feature adds no value to any analysis
df = df.drop(columns=["delivery_charges"], errors="ignore")
print(f"\nAFTER -- 'delivery_charges' column dropped.")
print(f"Remaining columns: {len(df.columns)}")

# %% [markdown]
# ## Challenge 10 -- Festival Name Nulls Are Informative (Not Missing Data)
#
# **Problem:** `festival_name` has 777,736 nulls -- but these aren't missing data.
# They mean "this order happened during a non-festival period". We should
# use this to engineer `is_festival_sale` correctly and create `sale_type`.

# %%
print(f"BEFORE -- festival_name nulls: {df_raw['festival_name'].isna().sum():,} ({df_raw['festival_name'].isna().mean()*100:.1f}%)")
print(f"Unique festivals: {df_raw['festival_name'].dropna().unique()}")

# %%
# FIX: Engineer meaningful columns from festival_name
# is_festival_sale: bool (True if festival_name is not null)
df["is_festival_sale"] = df["festival_name"].notna()

# sale_type: categorical
df["sale_type"] = df.apply(
    lambda row: (
        "festival" if row["is_festival_sale"]
        else "normal"
    ),
    axis=1
)

# Fill festival_name nulls with "No Festival" for display
df["festival_name"] = df["festival_name"].fillna("No Festival")

# AFTER
print(f"\nAFTER:")
print(f"  is_festival_sale True: {df['is_festival_sale'].sum():,} ({df['is_festival_sale'].mean()*100:.1f}%)")
print(f"  sale_type distribution:\n{df['sale_type'].value_counts().to_string()}")
print(f"  Top festivals:\n{df[df['is_festival_sale']]['festival_name'].value_counts().head(5).to_string()}")

# %% [markdown]
# ## Feature Engineering: Additional Derived Columns

# %%
# payment_category: group payment methods
PAYMENT_CAT = {
    "UPI": "Digital",
    "Debit Card": "Card",
    "Credit Card": "Card",
    "Net Banking": "Digital",
    "Wallet": "Digital",
    "COD": "Cash",
    "BNPL": "Credit",
}
df["payment_category"] = df["payment_method"].map(PAYMENT_CAT).fillna("Other")

# Drop source_year (duplicate of order_year, added during load)
df = df.drop(columns=["source_year"], errors="ignore")

print("Feature engineering complete:")
print(f"  payment_category: {df['payment_category'].value_counts().to_dict()}")

# %% [markdown]
# ## Final Summary: Before vs After Cleaning

# %%
print("=" * 60)
print("DATA CLEANING SUMMARY")
print("=" * 60)
print(f"\nRaw rows:          {len(df_raw):,}")
print(f"Clean rows:        {len(df):,}")
print(f"Removed:           {len(df_raw) - len(df):,} rows (duplicates)")
print(f"\nRaw columns:       {len(df_raw.columns)}")
print(f"Clean columns:     {len(df.columns)}")
print()
print("Cleaning actions taken:")
print("  1. original_price_inr : removed commas -> float")
print("  2. delivery_days      : parsed text/ranges -> float, capped at 30")
print("  3. is_prime_member etc: standardized 8 variants -> Python bool")
print("  4. customer_rating    : parsed 21 formats -> float, imputed with product median")
print("  5. order_date         : parsed 3 date formats -> datetime64")
print("  6. category           : unified 5 spelling variants -> 'Electronics'")
print("  7. customer_age_group : filled 135K nulls with tier-mode")
print("  8. Duplicates         : removed _DUP rows + exact duplicates")
print("  9. delivery_charges   : dropped (all-zero, no signal)")
print(" 10. festival_name      : nulls -> 'No Festival'; engineered is_festival_sale")
print()
print("New columns added:")
print("  rating_missing    : 1 if rating was originally null")
print("  payment_category  : Digital / Card / Cash / Credit")
print()

# Final null check
nulls = df.isna().sum()
nulls_remaining = nulls[nulls > 0]
if nulls_remaining.empty:
    print("Remaining nulls: NONE OK")
else:
    print("Remaining nulls:")
    print(nulls_remaining.to_string())

# %% [markdown]
# ## Save Cleaned Dataset

# %%
clean_path = PROCESSED_DIR / "cleaned_df_sales.csv"
df.to_csv(clean_path, index=False)
print(f"Saved cleaned dataset: {clean_path}")
print(f"  Shape: {df.shape}")
print(f"  Size: {clean_path.stat().st_size / 1e6:.1f} MB")

# Save product catalog (minimal cleaning needed)
prod_clean = df_products_raw.copy()
prod_clean.to_csv(PROCESSED_DIR / "cleaned_products.csv", index=False)
print(f"Saved product catalog: {PROCESSED_DIR / 'cleaned_products.csv'}")
print(f"  Shape: {prod_clean.shape}")

# %%
# Final dtypes overview
print("\n=== Final Column Dtypes ===")
print(df.dtypes.to_string())

# %% [markdown]
# ## ✅ Next Step
#
# Run `notebooks/02_eda.py` to see 20 analyses on this cleaned data.
# (Or run `notebooks/01_data_engineering.py` to load the cleaned data into PostgreSQL.)
