import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config import ARTIFACTS_DIR, MLFLOW_TRACKING_URI, RANDOM_STATE

logger = logging.getLogger(__name__)

# --- Diwali dates (custom holidays — never add_country_holidays + manual regressor together) ---
_DIWALI_DATES = pd.to_datetime([
    "2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07",
    "2019-10-27", "2020-11-14", "2021-11-04", "2022-10-24",
    "2023-11-12", "2024-11-01",
])
_DIWALI_HOLIDAYS = pd.DataFrame({
    "holiday":      "diwali",
    "ds":           _DIWALI_DATES,
    "lower_window": -7,
    "upper_window": 3,
})

# Minimum months of history required to detect yearly seasonality
_MIN_MONTHS = 12

# SQL to pull monthly subcategory revenue
_MONTHLY_QUERY = text("""
SELECT
    date_trunc('month', ft.order_date)::date AS month,
    dp.subcategory,
    SUM(ft.final_amount_inr)                 AS total_revenue
FROM fact_transactions ft
JOIN dim_products dp ON ft.product_id = dp.product_id
WHERE ft.order_date IS NOT NULL
  AND dp.subcategory IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, 2
""")


def slug_from_subcategory(s: str) -> str:
    """Canonical slug used in save/load/serve paths — must be identical in all 3 places."""
    return s.lower().replace(" ", "_").replace("/", "_")


def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Mean Absolute Percentage Error — robust to near-zero denominators."""
    total = np.sum(np.abs(y_true))
    if total == 0:
        return float("inf")
    return float(np.sum(np.abs(y_true - y_pred)) / total)


def _setup_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)


def _train_one_subcategory(
    monthly_df: pd.DataFrame,
    slug: str,
    models_dir: Path,
    parent_run_id: str,
) -> float | None:
    """
    Trains a Prophet model for one subcategory.

    Returns WMAPE, or None if skipped (insufficient data).
    Saves model to disk and logs to MLflow child run.
    """
    if len(monthly_df) < _MIN_MONTHS:
        logger.warning(
            "Skipping %s — only %d months of data (need >= %d for yearly seasonality)",
            slug, len(monthly_df), _MIN_MONTHS,
        )
        return None

    # Prophet requires columns named 'ds' and 'y'
    prophet_df = monthly_df[["month", "total_revenue"]].rename(
        columns={"month": "ds", "total_revenue": "y"}
    )
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    # Train/test: last 3 months held out for WMAPE evaluation
    n_test = 3
    train_df = prophet_df.iloc[:-n_test]
    test_df  = prophet_df.iloc[-n_test:]

    m = Prophet(
        holidays=_DIWALI_HOLIDAYS,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train_df)

    # Forecast over test horizon
    future = m.make_future_dataframe(periods=n_test, freq="MS")
    forecast = m.predict(future)
    y_pred = forecast.iloc[-n_test:]["yhat"].values
    y_true = test_df["y"].values
    wmape  = _wmape(y_true, y_pred)

    # Save model to disk FIRST (Rule 37), then log to MLflow child run
    model_path = models_dir / f"forecast_{slug}.json"
    with open(model_path, "w") as f:
        f.write(model_to_json(m))

    with mlflow.start_run(run_name=f"forecast_{slug}", nested=True, tags={"parent_run_id": parent_run_id}):
        mlflow.log_params({
            "subcategory": slug,
            "n_train_months": len(train_df),
            "n_test_months": n_test,
            "seasonality_mode": "multiplicative",
        })
        mlflow.log_metrics({"wmape": wmape, "n_train_months": len(train_df)})
        mlflow.log_artifact(str(model_path))

    logger.info("  %s — WMAPE=%.3f | trained on %d months", slug, wmape, len(train_df))
    return wmape


def train_forecast_models(engine: Engine) -> dict[str, float]:
    """
    Trains one Prophet model per subcategory with MLflow tracking.

    Skips subcategories with fewer than 12 months of data (warns, does not raise).
    Uses last 3 months as holdout for WMAPE evaluation.

    Returns:
        dict mapping slug → WMAPE for all successfully trained subcategories.
    """
    models_dir = ARTIFACTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading monthly subcategory revenue from PostgreSQL...")
    with engine.connect() as conn:
        df = pd.read_sql(_MONTHLY_QUERY, conn)

    if df.empty:
        raise RuntimeError("No monthly data returned — check ETL pipeline")

    subcategories = df["subcategory"].unique().tolist()
    logger.info("Found %d subcategories", len(subcategories))

    _setup_mlflow("amazon_forecasting")
    wmape_by_slug: dict[str, float] = {}
    trained_slugs: list[str] = []

    with mlflow.start_run(run_name="forecast_models") as parent_run:
        mlflow.log_param("n_subcategories_attempted", len(subcategories))

        for subcat in sorted(subcategories):
            slug       = slug_from_subcategory(subcat)
            monthly_df = (
                df[df["subcategory"] == subcat]
                .sort_values("month")
                .reset_index(drop=True)
            )
            wmape = _train_one_subcategory(
                monthly_df, slug, models_dir, parent_run.info.run_id
            )
            if wmape is not None:
                wmape_by_slug[slug]  = wmape
                trained_slugs.append(slug)

        # Summary metrics
        if wmape_by_slug:
            mlflow.log_metrics({
                "n_subcategories_trained": len(trained_slugs),
                "mean_wmape":  float(np.mean(list(wmape_by_slug.values()))),
                "median_wmape": float(np.median(list(wmape_by_slug.values()))),
            })

        # Save slug index to disk FIRST, then log
        slugs_path = models_dir / "forecast_slugs.json"
        with open(slugs_path, "w") as f:
            json.dump({"slugs": trained_slugs}, f, indent=2)
        mlflow.log_artifact(str(slugs_path))

    logger.info(
        "Forecast training complete — %d/%d subcategories | mean WMAPE=%.3f",
        len(trained_slugs), len(subcategories),
        np.mean(list(wmape_by_slug.values())) if wmape_by_slug else float("nan"),
    )
    return wmape_by_slug


def load_forecast_model(subcategory: str) -> Prophet:
    """Loads a trained Prophet model for the given subcategory slug."""
    slug       = slug_from_subcategory(subcategory)
    model_path = ARTIFACTS_DIR / "models" / f"forecast_{slug}.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No forecast model for subcategory '{subcategory}' (slug='{slug}'). "
            "Run train_forecast_models() first."
        )
    with open(model_path) as f:
        m = model_from_json(f.read())
    return m


def predict_forecast(subcategory: str, periods: int = 3) -> pd.DataFrame:
    """
    Generates a forward forecast for the given subcategory.

    Returns a DataFrame with columns: ds, yhat, yhat_lower, yhat_upper.
    Only future rows (beyond training data) are returned.
    """
    m = load_forecast_model(subcategory)
    future   = m.make_future_dataframe(periods=periods, freq="MS")
    forecast = m.predict(future)
    future_rows = forecast.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    future_rows["subcategory"] = subcategory
    return future_rows.reset_index(drop=True)


def list_trained_subcategories() -> list[str]:
    """Returns list of slugs for all trained forecast models."""
    slugs_path = ARTIFACTS_DIR / "models" / "forecast_slugs.json"
    if not slugs_path.exists():
        return []
    with open(slugs_path) as f:
        return json.load(f)["slugs"]
