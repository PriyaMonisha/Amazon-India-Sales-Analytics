"""
All 10 data cleaning challenges for Amazon India Sales data.
Challenge list:
  1. Date standardization
  2. Price column cleaning
  3. Rating standardization
  4. City name standardization
  5. Boolean column cleaning
  6. Category consolidation
  7. Delivery days cleaning
  8. Duplicate detection and removal
  9. Price outlier capping (IQR per subcategory)
  10. Payment method standardization
"""
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# City name mapping (Bengaluru → Bangalore, etc.)
# ──────────────────────────────────────────────────────────────────────────────
CITY_ALIASES: dict[str, str] = {
    "bengaluru": "Bangalore",
    "banglore": "Bangalore",
    "bangaluru": "Bangalore",
    "bombay": "Mumbai",
    "new delhi": "Delhi",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "mysore": "Mysuru",
    "mangalore": "Mangaluru",
}

CATEGORY_ALIASES: dict[str, str] = {
    "electronics & accessories": "Electronics",
    "electronic": "Electronics",
    "electronics": "Electronics",
}

PAYMENT_MAP: dict[str, str] = {
    "phonepe": "UPI",
    "googlepay": "UPI",
    "google pay": "UPI",
    "gpay": "UPI",
    "bhim": "UPI",
    "paytm upi": "UPI",
    "upi": "UPI",
    "credit card": "Credit Card",
    "credit_card": "Credit Card",
    "cc": "Credit Card",
    "creditcard": "Credit Card",
    "debit card": "Debit Card",
    "debit_card": "Debit Card",
    "dc": "Debit Card",
    "debitcard": "Debit Card",
    "cash on delivery": "COD",
    "cod": "COD",
    "c.o.d": "COD",
    "c.o.d.": "COD",
    "net banking": "Net Banking",
    "netbanking": "Net Banking",
    "net_banking": "Net Banking",
    "wallet": "Wallet",
    "emi": "EMI",
    "bnpl": "BNPL",
    "buy now pay later": "BNPL",
}

PAYMENT_CATEGORIES: dict[str, str] = {
    "UPI": "Digital",
    "Credit Card": "Card",
    "Debit Card": "Card",
    "Net Banking": "Digital",
    "Wallet": "Digital",
    "COD": "Cash",
    "EMI": "Credit",
    "BNPL": "Credit",
}


def _parse_price(val: object) -> float | None:
    """Strip ₹, commas, spaces; return float or None."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "price on request", "-"):
        return None
    s = re.sub(r"[₹,\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _parse_rating(val: object) -> float | None:
    """Parse '4 stars', '3/5', '2.5/5.0', or plain float."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ("nan", "none", "null", ""):
        return None
    # "4 stars" → 4.0
    m = re.match(r"^(\d+(?:\.\d+)?)\s*stars?$", s)
    if m:
        return float(m.group(1))
    # "3/5" or "2.5/5.0" → normalise to /5 scale
    m = re.match(r"^(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$", s)
    if m:
        num, denom = float(m.group(1)), float(m.group(2))
        if denom > 0:
            return round(num * 5.0 / denom, 2)
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(val: object) -> pd.Timestamp | None:
    """Try multiple date formats; return Timestamp or None."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%m/%d/%Y"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            pass
    try:
        return pd.Timestamp(s)
    except Exception:
        return None


def _to_bool(val: object) -> bool | None:
    """Map Yes/No/1/0/TRUE/FALSE/True/False → Python bool."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ("true", "yes", "1", "y"):
        return True
    if s in ("false", "no", "0", "n"):
        return False
    return None


def _standardize_city(val: object) -> str:
    """Lowercase → check alias dict → return standardized name."""
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip()
    key = s.lower()
    return CITY_ALIASES.get(key, s.title())


def _standardize_payment(val: object) -> str:
    """Map all payment variants to canonical names."""
    if pd.isna(val):
        return "Unknown"
    key = str(val).strip().lower()
    return PAYMENT_MAP.get(key, str(val).strip().title())


# ──────────────────────────────────────────────────────────────────────────────
# Main transform function
# ──────────────────────────────────────────────────────────────────────────────

def clean_sales(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply all 10 cleaning challenges to the raw sales DataFrame.

    Returns: (cleaned_df, cleaning_log)
    cleaning_log tracks counts for data observability.
    """
    log: dict[str, int] = {
        "input_rows": len(df),
        "dates_fixed": 0,
        "dates_sentineled": 0,
        "prices_imputed": 0,
        "ratings_imputed": 0,
        "cities_standardized": 0,
        "payment_standardized": 0,
        "duplicates_removed": 0,
        "outliers_capped": 0,
        "delivery_fixed": 0,
    }

    df = df.copy()

    # ── Challenge 1: Date standardization ────────────────────────────────────
    if "order_date" in df.columns:
        original = df["order_date"].copy()
        df["order_date"] = df["order_date"].apply(_parse_date)
        fixed = df["order_date"].notna() & original.notna() & (df["order_date"].astype(str) != original.astype(str))
        log["dates_fixed"] = int(fixed.sum())

        # Map NaT → sentinel '1900-01-01' (star schema FK safety)
        sentinel_mask = df["order_date"].isna()
        log["dates_sentineled"] = int(sentinel_mask.sum())
        df["order_date"] = df["order_date"].fillna(pd.Timestamp("1900-01-01"))
        df["order_date"] = pd.to_datetime(df["order_date"])

    # ── Challenge 2: Price column cleaning ───────────────────────────────────
    price_cols = ["original_price_inr", "discounted_price_inr", "subtotal_inr",
                  "final_amount_inr", "delivery_charges"]
    for col in price_cols:
        if col not in df.columns:
            continue
        original_nulls = df[col].isna().sum()
        df[col] = df[col].apply(_parse_price)
        new_nulls = df[col].isna().sum()
        if new_nulls > original_nulls:
            log["prices_imputed"] += int(new_nulls - original_nulls)

    # Impute missing prices with subcategory median
    if "original_price_inr" in df.columns and "subcategory" in df.columns:
        medians = df.groupby("subcategory")["original_price_inr"].transform("median")
        null_mask = df["original_price_inr"].isna()
        df.loc[null_mask, "original_price_inr"] = medians[null_mask]
        log["prices_imputed"] += int(null_mask.sum())

    # ── Challenge 3: Rating standardization ──────────────────────────────────
    for rating_col in ["customer_rating", "product_rating"]:
        if rating_col not in df.columns:
            continue
        df[rating_col] = df[rating_col].apply(_parse_rating)
        null_mask = df[rating_col].isna()
        if null_mask.any():
            if "product_id" in df.columns:
                medians = df.groupby("product_id")[rating_col].transform("median")
            else:
                medians = df[rating_col].median()
            df.loc[null_mask, rating_col] = medians[null_mask] if isinstance(medians, pd.Series) else medians
            log["ratings_imputed"] += int(null_mask.sum())

    # ── Challenge 4: City standardization ────────────────────────────────────
    if "customer_city" in df.columns:
        original_cities = df["customer_city"].copy()
        df["customer_city"] = df["customer_city"].apply(_standardize_city)
        changed = df["customer_city"] != original_cities.fillna("Unknown")
        log["cities_standardized"] = int(changed.sum())

    # ── Challenge 5: Boolean standardization ─────────────────────────────────
    bool_cols = ["is_prime_member", "is_prime_eligible", "is_festival_sale"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(_to_bool)

    # ── Challenge 6: Category consolidation ──────────────────────────────────
    if "category" in df.columns:
        df["category"] = (
            df["category"]
            .str.strip()
            .str.lower()
            .map(lambda v: CATEGORY_ALIASES.get(v, v) if isinstance(v, str) else v)
        )

    # ── Challenge 7: Delivery days cleaning ──────────────────────────────────
    if "delivery_days" in df.columns:
        def clean_delivery(v: object) -> float | None:
            if pd.isna(v):
                return None
            s = str(v).strip().lower()
            if s in ("same day", "same-day", "0 days"):
                return 0.0
            # Range "1-2 days" → mean
            m = re.match(r"(\d+)\s*[-–]\s*(\d+)", s)
            if m:
                return (float(m.group(1)) + float(m.group(2))) / 2
            try:
                val = float(re.sub(r"[^\d.]", "", s))
                return val if 0 <= val <= 30 else None  # cap IQR-style at 30
            except ValueError:
                return None

        original_nulls = df["delivery_days"].isna().sum()
        df["delivery_days"] = df["delivery_days"].apply(clean_delivery)
        log["delivery_fixed"] = int(df["delivery_days"].isna().sum() - original_nulls)
        # Negative values → NaN
        neg_mask = df["delivery_days"] < 0
        df.loc[neg_mask, "delivery_days"] = np.nan

    # ── Challenge 8: Duplicate detection and removal ──────────────────────────
    if "transaction_id" in df.columns:
        # Remove rows where transaction_id ends with _DUP
        dup_mask = df["transaction_id"].str.endswith("_DUP", na=False)
        exact_dups = df.duplicated(keep="first")
        remove_mask = dup_mask | exact_dups
        log["duplicates_removed"] = int(remove_mask.sum())
        df = df[~remove_mask].reset_index(drop=True)

    # ── Challenge 9: Price outlier capping (IQR per subcategory) ─────────────
    if "original_price_inr" in df.columns and "subcategory" in df.columns:
        capped = 0
        for subcat, grp in df.groupby("subcategory"):
            prices = grp["original_price_inr"].dropna()
            q1, q3 = prices.quantile(0.25), prices.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 3 * iqr
            # Domain cap: electronics ≤ ₹2L
            domain_cap = 200_000.0
            cap = min(upper, domain_cap)
            mask = (df["subcategory"] == subcat) & (df["original_price_inr"] > cap)
            if mask.any():
                df.loc[mask, "original_price_inr"] = cap
                capped += int(mask.sum())
        log["outliers_capped"] = capped

    # ── Challenge 10: Payment method standardization ──────────────────────────
    if "payment_method" in df.columns:
        original_payment = df["payment_method"].copy()
        df["payment_method"] = df["payment_method"].apply(_standardize_payment)
        changed = df["payment_method"] != original_payment.fillna("Unknown")
        log["payment_standardized"] = int(changed.sum())

    # ── Engineering: derived columns ─────────────────────────────────────────
    if "festival_name" in df.columns and "is_festival_sale" not in df.columns:
        df["is_festival_sale"] = df["festival_name"].notna() & (df["festival_name"].str.strip() != "")

    if "customer_rating" in df.columns:
        df["rating_missing"] = df["customer_rating"].isna().astype(int)

    if "payment_method" in df.columns:
        df["payment_category"] = df["payment_method"].map(PAYMENT_CATEGORIES).fillna("Other")

    if "is_festival_sale" in df.columns:
        def sale_type(row: pd.Series) -> str:
            if row.get("is_festival_sale", False):
                return "festival"
            return "normal"
        df["sale_type"] = df.apply(sale_type, axis=1)

    # Drop always-zero or single-value columns (detected dynamically)
    cols_to_drop = []
    if "category" in df.columns and df["category"].nunique(dropna=False) == 1:
        cols_to_drop.append("category")
    if "delivery_charges" in df.columns and df["delivery_charges"].fillna(0).eq(0).all():
        cols_to_drop.append("delivery_charges")
    # Drop duplicate 'year' col if order_year already exists
    if "year" in df.columns and "order_year" in df.columns:
        cols_to_drop.append("year")
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped always-constant/duplicate cols: {cols_to_drop}")

    log["output_rows"] = len(df)
    logger.info(f"Cleaning complete: {log['input_rows']:,} → {log['output_rows']:,} rows | {log}")
    return df, log


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning for the product catalog."""
    df = df.copy()
    if "category" in df.columns:
        df["category"] = df["category"].str.strip().str.lower().map(
            lambda v: CATEGORY_ALIASES.get(v, v) if isinstance(v, str) else v
        )
    if "is_prime_eligible" in df.columns:
        df["is_prime_eligible"] = df["is_prime_eligible"].apply(_to_bool)
    return df


def build_cleaning_log(log: dict, artifacts_dir: Path) -> None:
    """
    Save cleaning log with timestamp (not just date — avoids same-day overwrites).
    Also maintains rolling summary of last 30 runs.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log["run_timestamp"] = timestamp

    drift_dir = artifacts_dir / "drift"
    drift_dir.mkdir(parents=True, exist_ok=True)

    # Individual run log
    run_log_path = drift_dir / f"cleaning_log_{timestamp}.json"
    run_log_path.write_text(json.dumps(log, indent=2))

    # Rolling summary (last 30 runs)
    summary_path = drift_dir / "cleaning_summary.json"
    history: list[dict] = json.loads(summary_path.read_text()) if summary_path.exists() else []
    history.append(log)
    summary_path.write_text(json.dumps(history[-30:], indent=2))

    logger.info(f"Cleaning log saved: {run_log_path.name}")
