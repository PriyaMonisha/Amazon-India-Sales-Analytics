"""
Data validation suite — same 7 expectations as Great Expectations, implemented in pure pandas.

Why pure pandas instead of GE:
  - GE 0.18.19 requires ipywidgets → jupyterlab-widgets, which have file paths that
    exceed Windows MAX_PATH (260 chars) when venv is in a long directory path.
  - Pure pandas validation has zero extra dependencies, is faster, and is portable.
  - The logic is identical; this is more honest than hiding it behind a framework.

If any expectation fails → raises RuntimeError.
In Airflow: RuntimeError marks the task as failed → downstream DAGs are blocked.
"""
import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

VALID_PAYMENT_METHODS = frozenset([
    "UPI", "Credit Card", "Debit Card", "COD",
    "Net Banking", "Wallet", "EMI", "BNPL", "Unknown",
])

MIN_ROWS_SALES = 1_000_000
MAX_ROWS_SALES = 1_300_000
MIN_ROWS_PRODUCTS = 1_500
MAX_ROWS_PRODUCTS = 3_000


@dataclass
class ExpectationResult:
    name: str
    success: bool
    failure_count: int
    detail: str = ""


def _check(name: str, passed: bool, failure_count: int = 0, detail: str = "") -> ExpectationResult:
    r = ExpectationResult(name=name, success=passed, failure_count=failure_count, detail=detail)
    if not passed:
        logger.warning(f"  FAILED [{name}]: {failure_count} violations — {detail}")
    return r


def validate_sales(df: pd.DataFrame) -> bool:
    """
    Validate the cleaned sales DataFrame against 7 expectations.
    Returns True if all pass. Raises RuntimeError if any fail.
    """
    results: list[ExpectationResult] = []

    # 1. final_amount_inr between 1 and 500,000
    if "final_amount_inr" in df.columns:
        col = df["final_amount_inr"].dropna()
        bad = (~col.between(1, 500_000)).sum()
        results.append(_check("price_in_range_1_500k", bad == 0, int(bad), f"{bad} values outside [1, 500000]"))

    # 2. customer_rating between 0 and 5 (95% compliance — allows some nulls)
    if "customer_rating" in df.columns:
        col = df["customer_rating"].dropna()
        bad = (~col.between(0.0, 5.0)).sum()
        compliance = 1 - bad / max(len(col), 1)
        results.append(_check("rating_0_to_5_mostly", compliance >= 0.95, int(bad),
                               f"compliance={compliance:.3f} (need ≥ 0.95)"))

    # 3. delivery_days between 0 and 30 (95% compliance — allows some nulls)
    if "delivery_days" in df.columns:
        col = df["delivery_days"].dropna()
        bad = (~col.between(0, 30)).sum()
        compliance = 1 - bad / max(len(col), 1)
        results.append(_check("delivery_days_0_to_30_mostly", compliance >= 0.95, int(bad),
                               f"compliance={compliance:.3f} (need ≥ 0.95)"))

    # 4. payment_method in valid set
    if "payment_method" in df.columns:
        invalid = df["payment_method"].dropna()
        bad_vals = (~invalid.isin(VALID_PAYMENT_METHODS)).sum()
        if bad_vals:
            unknowns = invalid[~invalid.isin(VALID_PAYMENT_METHODS)].unique()[:5]
            results.append(_check("payment_method_in_set", False, int(bad_vals), f"unknown values: {list(unknowns)}"))
        else:
            results.append(_check("payment_method_in_set", True))

    # 5. transaction_id is unique
    if "transaction_id" in df.columns:
        dupes = df["transaction_id"].duplicated().sum()
        results.append(_check("transaction_id_unique", dupes == 0, int(dupes), f"{dupes} duplicates"))

    # 6. customer_id not null
    if "customer_id" in df.columns:
        nulls = df["customer_id"].isna().sum()
        results.append(_check("customer_id_not_null", nulls == 0, int(nulls), f"{nulls} nulls"))

    # 7. Row count in expected range
    n = len(df)
    results.append(_check("row_count_in_range", MIN_ROWS_SALES <= n <= MAX_ROWS_SALES,
                           0 if MIN_ROWS_SALES <= n <= MAX_ROWS_SALES else 1,
                           f"got {n:,}, expected [{MIN_ROWS_SALES:,}, {MAX_ROWS_SALES:,}]"))

    failed = [r for r in results if not r.success]
    if failed:
        names = [r.name for r in failed]
        msg = f"Validation failed: {len(failed)} checks — {names}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(f"Validation passed: {len(results)} checks, {len(df):,} rows.")
    return True


def validate_products(df: pd.DataFrame) -> bool:
    """Validate the product catalog."""
    results: list[ExpectationResult] = []

    if "product_id" in df.columns:
        dupes = df["product_id"].duplicated().sum()
        results.append(_check("product_id_unique", dupes == 0, int(dupes)))
        nulls = df["product_id"].isna().sum()
        results.append(_check("product_id_not_null", nulls == 0, int(nulls)))

    n = len(df)
    results.append(_check("product_row_count", MIN_ROWS_PRODUCTS <= n <= MAX_ROWS_PRODUCTS,
                           detail=f"got {n:,}"))

    failed = [r for r in results if not r.success]
    if failed:
        raise RuntimeError(f"Product catalog validation failed: {[r.name for r in failed]}")

    logger.info(f"Product catalog validation passed: {n:,} products.")
    return True
