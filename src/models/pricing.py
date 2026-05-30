import json
import logging
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sqlalchemy import text
from sqlalchemy.engine import Engine

import config
from config import ARTIFACTS_DIR, RANDOM_STATE
from src.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

# Festival months: Oct (10) and Nov (11) — Diwali/Navratri peak season
_FESTIVAL_MONTHS = {10, 11}

# SQL: monthly aggregation per subcategory × price bucket bin
_PRICING_QUERY = text("""
SELECT
    dp.subcategory,
    date_trunc('month', ft.order_date)::date    AS month,
    ft.final_amount_inr                         AS price_inr,
    ft.mrp_inr,
    COUNT(*)                                    AS order_count
FROM fact_transactions ft
JOIN dim_products dp ON ft.product_id = dp.product_id
WHERE ft.order_date IS NOT NULL
  AND ft.final_amount_inr > 0
  AND ft.mrp_inr > 0
  AND dp.subcategory IS NOT NULL
GROUP BY 1, 2, ft.final_amount_inr, ft.mrp_inr
""")


def _make_price_bucket(prices: pd.Series, n_bins: int = 10) -> pd.Series:
    """
    Assigns price bucket index (0..n_bins-1).

    Uses pd.qcut(duplicates='drop') to handle uneven distributions.
    Falls back to rank-based pd.cut if too few unique prices for qcut.
    """
    try:
        return pd.qcut(prices, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.cut(
            prices.rank(pct=True),
            bins=n_bins,
            labels=False,
            include_lowest=True,
        )


def _build_features(df: pd.DataFrame, encoder: OrdinalEncoder | None = None) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """
    Builds model features from raw pricing data.

    Returns (feature_df, fitted_encoder).
    Pass a pre-fitted encoder for inference (not None).
    """
    agg = (
        df.groupby(["subcategory", "month"])
        .agg(
            avg_price_inr=("price_inr", "mean"),
            order_count=("order_count", "sum"),
            avg_mrp=("mrp_inr", "mean"),
        )
        .reset_index()
    )

    agg["month"] = pd.to_datetime(agg["month"])
    agg["month_of_year"]    = agg["month"].dt.month.astype(float)
    agg["is_festival_month"] = agg["month_of_year"].isin(_FESTIVAL_MONTHS).astype(float)
    agg["log_avg_price"]    = np.log1p(agg["avg_price_inr"])
    agg["log_order_count"]  = np.log1p(agg["order_count"])

    # Subcategory encoding
    if encoder is None:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        agg["subcategory_encoded"] = encoder.fit_transform(agg[["subcategory"]])
    else:
        agg["subcategory_encoded"] = encoder.transform(agg[["subcategory"]])

    return agg, encoder


_FEATURE_COLS = ["log_avg_price", "month_of_year", "is_festival_month", "subcategory_encoded"]
_TARGET_COL   = "log_order_count"


def train_pricing_model(engine: Engine) -> dict[str, Any]:
    """
    Trains an XGBoost regressor to model price → demand elasticity.

    Target: log1p(order_count) per (subcategory × month).
    Features: log1p(avg_price_inr), month_of_year, is_festival_month, subcategory_encoded.

    This is a log-log model: the coefficient of log_avg_price approximates price elasticity.

    Returns dict with keys: model, encoder, r2, rmse, feature_names.
    """
    models_dir = ARTIFACTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading pricing data from PostgreSQL...")
    with engine.connect() as conn:
        df = pd.read_sql(_PRICING_QUERY, conn)

    if df.empty:
        raise RuntimeError("No pricing data returned — check ETL pipeline")

    logger.info("Loaded %d pricing rows", len(df))

    # Build features
    feat_df, encoder = _build_features(df)
    feat_df = feat_df.dropna(subset=_FEATURE_COLS + [_TARGET_COL])

    # Train/test split: last 3 months as test
    feat_df = feat_df.sort_values("month").reset_index(drop=True)
    cutoff   = feat_df["month"].max() - pd.DateOffset(months=3)
    train_df = feat_df[feat_df["month"] <= cutoff]
    test_df  = feat_df[feat_df["month"] > cutoff]

    X_train = train_df[_FEATURE_COLS].values
    y_train = train_df[_TARGET_COL].values
    X_test  = test_df[_FEATURE_COLS].values
    y_test  = test_df[_TARGET_COL].values

    setup_mlflow("amazon_pricing")
    # No Optuna: log-log model at monthly aggregated grain is insensitive to tree depth.
    # Add Optuna if grain drops to weekly or feature set grows beyond 6.
    params = {
        "n_estimators":     200,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "random_state":     RANDOM_STATE,
        "eval_metric":      "rmse",
        "tree_method":      "hist",
    }

    with mlflow.start_run(run_name="pricing_elasticity") as run:
        mlflow.log_params(params)
        mlflow.log_params({
            "train_rows": len(X_train),
            "test_rows":  len(X_test),
            "features":   ",".join(_FEATURE_COLS),
        })

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)
        r2   = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        mlflow.log_metrics({"r2": r2, "rmse": rmse})

        if r2 < config.PRICING_MIN_R2:
            logger.warning(
                "Pricing model R²=%.3f below threshold %.3f — predictions may be unreliable",
                r2, config.PRICING_MIN_R2,
            )

        logger.info("Pricing model — R²=%.3f | RMSE=%.4f", r2, rmse)

        # Save quality gate (load_pricing_model refuses to load if meets_threshold=False)
        quality = {
            "r2":              float(r2),
            "rmse":            float(rmse),
            "meets_threshold": r2 >= config.PRICING_MIN_R2,
        }
        quality_path = models_dir / "pricing_quality.json"
        quality_path.write_text(json.dumps(quality, indent=2))
        mlflow.log_artifact(str(quality_path))

        # Save to disk FIRST, then log artifacts
        model_path   = models_dir / "pricing_model.json"
        encoder_path = models_dir / "pricing_encoder.pkl"

        model.save_model(str(model_path))
        joblib.dump(encoder, encoder_path)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(encoder_path))

        # Feature metadata
        meta = {
            "feature_names": _FEATURE_COLS,
            "target":        _TARGET_COL,
            "r2":            r2,
            "rmse":          rmse,
            "run_id":        run.info.run_id,
        }
        meta_path = models_dir / "pricing_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact(str(meta_path))

    return {
        "model":         model,
        "encoder":       encoder,
        "r2":            r2,
        "rmse":          rmse,
        "feature_names": _FEATURE_COLS,
    }


def load_pricing_model() -> dict[str, Any]:
    """Loads pricing model and encoder from disk. Raises RuntimeError if quality gate fails."""
    models_dir = ARTIFACTS_DIR / "models"

    q_path = models_dir / "pricing_quality.json"
    if q_path.exists():
        q = json.loads(q_path.read_text())
        if not q["meets_threshold"]:
            raise RuntimeError(
                f"Pricing model R²={q['r2']:.3f} is below production threshold "
                f"{config.PRICING_MIN_R2}. Retrain required."
            )

    model = xgb.XGBRegressor()
    model.load_model(str(models_dir / "pricing_model.json"))
    encoder: OrdinalEncoder = joblib.load(models_dir / "pricing_encoder.pkl")

    with open(models_dir / "pricing_meta.json") as f:
        meta = json.load(f)

    logger.info("Pricing model loaded from %s", models_dir)
    return {"model": model, "encoder": encoder, "feature_names": meta["feature_names"]}


def predict_optimal_price(
    subcategory: str,
    current_price: float,
    month: int | None = None,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Estimates demand at current price and suggests a price range.

    Uses log-log elasticity: elasticity ≈ d(log_quantity) / d(log_price).
    Pass `bundle` from app.state for zero-disk-I/O inference; omit for CLI use.
    Returns dict with current_demand_estimate, price_suggestions.
    """
    if bundle is None:
        bundle = load_pricing_model()
    model: xgb.XGBRegressor = bundle["model"]
    encoder: OrdinalEncoder = bundle["encoder"]

    if month is None:
        month = pd.Timestamp.now().month

    is_festival = 1.0 if month in _FESTIVAL_MONTHS else 0.0
    subcat_enc  = encoder.transform([[subcategory]])[0][0]
    log_price   = np.log1p(current_price)

    base_features = np.array([[log_price, float(month), is_festival, subcat_enc]])
    log_demand    = float(model.predict(base_features)[0])
    demand_est    = float(np.expm1(log_demand))

    # Evaluate demand at ±10%, ±20% price points
    suggestions = []
    for delta_pct in [-20, -10, 0, 10, 20]:
        adj_price     = current_price * (1 + delta_pct / 100)
        adj_features  = np.array([[np.log1p(adj_price), float(month), is_festival, subcat_enc]])
        adj_log_d     = float(model.predict(adj_features)[0])
        adj_demand    = float(np.expm1(adj_log_d))
        revenue_est   = adj_price * adj_demand
        suggestions.append({
            "price_inr":      round(adj_price, 2),
            "delta_pct":      delta_pct,
            "demand_estimate": round(adj_demand, 1),
            "revenue_estimate": round(revenue_est, 2),
        })

    return {
        "subcategory":            subcategory,
        "current_price_inr":      current_price,
        "current_demand_estimate": round(demand_est, 1),
        "month":                  month,
        "is_festival_month":      bool(is_festival),
        "price_suggestions":      suggestions,
    }
