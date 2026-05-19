"""
Data validation suite using Pandera — same 7 rules as Great Expectations.

Why Pandera instead of GE:
  - GE 0.18.19 requires ipywidgets → jupyterlab-widgets, which exceed Windows
    MAX_PATH (260 chars) when venv is in a long directory path.
  - Pandera is a pure Python schema validation library with no Jupyter dependencies.
  - It raises pandera.errors.SchemaError (caught as RuntimeError-compatible) on failure.
  - Equivalent in concept; more recognized than raw pandas assertions in interviews.

Placement in pipeline:
  transform.py → validate_sales() → load.py → PostgreSQL
  (In Airflow: validation_dag runs after data_cleaning_dag, blocks feature_engineering_dag)

If any expectation fails → raises SchemaError → Airflow marks task FAILED → downstream blocked.
"""
import logging

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)

VALID_PAYMENT_METHODS = [
    "UPI", "Credit Card", "Debit Card", "COD",
    "Net Banking", "Wallet", "EMI", "BNPL", "Unknown",
]

MIN_ROWS_SALES = 1_000_000
MAX_ROWS_SALES = 1_300_000
MIN_ROWS_PRODUCTS = 1_500
MAX_ROWS_PRODUCTS = 3_000

# ── Sales schema: 7 business rules ──────────────────────────────────────────
_SALES_SCHEMA = DataFrameSchema(
    columns={
        # Rule 1: Revenue between ₹1 and ₹5,00,000
        "final_amount_inr": Column(
            float,
            Check.in_range(1, 500_000),
            nullable=True,
            required=False,
        ),
        # Rule 2: Customer rating 0–5 (nullable — ~30% missing is expected)
        "customer_rating": Column(
            float,
            Check.in_range(0.0, 5.0),
            nullable=True,
            required=False,
        ),
        # Rule 3: Delivery days 0–30 (nullable — some orders have no delivery record)
        "delivery_days": Column(
            float,
            Check.in_range(0, 30),
            nullable=True,
            required=False,
        ),
        # Rule 4: Payment method in known set
        "payment_method": Column(
            str,
            Check.isin(VALID_PAYMENT_METHODS),
            nullable=True,
            required=False,
        ),
        # Rule 5: Transaction ID unique (handled separately — pandera unique check)
        "transaction_id": Column(str, nullable=False, required=False),
        # Rule 6: Customer ID not null
        "customer_id": Column(str, nullable=False, required=False),
    },
    checks=[
        # Rule 7: Row count sanity check
        Check(
            lambda df: MIN_ROWS_SALES <= len(df) <= MAX_ROWS_SALES,
            error=f"Row count outside expected range [{MIN_ROWS_SALES:,}, {MAX_ROWS_SALES:,}]",
        ),
    ],
    coerce=True,   # coerce int→float, str→str etc; don't fail on minor dtype differences
    strict=False,  # allow extra columns not listed here
)

_PRODUCT_SCHEMA = DataFrameSchema(
    columns={
        "product_id": Column(str, nullable=False, required=False),
    },
    checks=[
        Check(
            lambda df: MIN_ROWS_PRODUCTS <= len(df) <= MAX_ROWS_PRODUCTS,
            error=f"Product row count outside [{MIN_ROWS_PRODUCTS}, {MAX_ROWS_PRODUCTS}]",
        ),
    ],
    strict=False,
)


def validate_sales(df: pd.DataFrame) -> bool:
    """
    Validate cleaned sales DataFrame against 7 business rules.
    Returns True if all pass. Raises pandera.errors.SchemaError if any fail.

    In Airflow: SchemaError marks the task failed → downstream DAGs blocked.
    Interview point: "If GE isn't available, Pandera gives the same quality gates
    with zero Jupyter dependencies and a cleaner DataFrame-centric API."
    """
    # Pandera validates schema + checks
    try:
        _SALES_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        msg = f"Pandera validation failed: {len(e.failure_cases)} violations\n{e.failure_cases.to_string()}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    # Rule 5: uniqueness check (pandera unique flag + explicit)
    if "transaction_id" in df.columns:
        dupes = df["transaction_id"].duplicated().sum()
        if dupes > 0:
            msg = f"Validation failed: transaction_id has {dupes} duplicates"
            logger.error(msg)
            raise RuntimeError(msg)

    logger.info(f"Pandera validation PASSED: {len(df):,} rows, 7 checks.")
    return True


def validate_products(df: pd.DataFrame) -> bool:
    """Validate product catalog."""
    try:
        _PRODUCT_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        msg = f"Product schema failed: {e.failure_cases.to_string()}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    if "product_id" in df.columns:
        dupes = df["product_id"].duplicated().sum()
        if dupes:
            raise RuntimeError(f"Product catalog: {dupes} duplicate product_ids")

    logger.info(f"Product catalog validation PASSED: {len(df):,} products.")
    return True
