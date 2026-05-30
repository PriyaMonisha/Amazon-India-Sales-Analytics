import logging
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config import ARTIFACTS_DIR, RANDOM_STATE
from src.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

# Contamination rate: expected fraction of anomalies in the dataset
_CONTAMINATION = 0.02

# SQL: order-level features for anomaly detection
_ANOMALY_QUERY = text("""
SELECT
    ft.transaction_id,
    ft.final_amount_inr,
    ft.mrp_inr,
    ft.delivery_days,
    CASE WHEN ft.return_status = 'Returned' THEN 1 ELSE 0 END AS is_return
FROM fact_transactions ft
WHERE ft.final_amount_inr IS NOT NULL
  AND ft.mrp_inr > 0
  AND ft.delivery_days IS NOT NULL
""")

_FEATURE_COLS = ["final_amount_inr", "discount_pct", "delivery_days", "is_return"]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds discount_pct and selects feature columns."""
    df = df.copy()
    df["discount_pct"] = np.clip(
        1.0 - df["final_amount_inr"] / df["mrp_inr"], 0.0, 1.0
    )
    return df[["transaction_id"] + _FEATURE_COLS]


def train_anomaly_model(engine: Engine) -> dict[str, Any]:
    """
    Trains IsolationForest for order-level anomaly detection.

    Features: final_amount_inr, discount_pct, delivery_days, is_return.
    Labels: -1 = anomaly, 1 = normal (IsolationForest convention).
    Output columns added by detect_anomalies(): is_anomaly (bool), anomaly_score (float).

    Artifacts:
        artifacts/models/anomaly_model.pkl   (IsolationForest, joblib)
        artifacts/models/anomaly_scaler.pkl  (StandardScaler, joblib)
    """
    models_dir = ARTIFACTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading anomaly detection data from PostgreSQL...")
    with engine.connect() as conn:
        df = pd.read_sql(_ANOMALY_QUERY, conn)

    if df.empty:
        raise RuntimeError("No anomaly data returned — check ETL pipeline")

    logger.info("Loaded %d orders for anomaly detection", len(df))

    feat_df = _build_features(df)
    feat_df = feat_df.dropna(subset=_FEATURE_COLS).reset_index(drop=True)

    X = feat_df[_FEATURE_COLS].values

    # Scale features before IsolationForest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # IsolationForest
    iso = IsolationForest(
        contamination=_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=100,
        n_jobs=2,    # avoid starving co-located Docker services; -1 uses all cores
    )
    labels  = iso.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
    scores  = iso.decision_function(X_scaled)  # higher = more normal

    n_anomalies  = int((labels == -1).sum())
    anomaly_rate = float(n_anomalies / len(labels))
    logger.info(
        "Anomaly detection: %d anomalies (%.2f%%) in %d orders",
        n_anomalies, anomaly_rate * 100, len(labels),
    )

    setup_mlflow("amazon_anomaly")
    with mlflow.start_run(run_name="isolation_forest_anomaly") as run:
        mlflow.log_params({
            "contamination": _CONTAMINATION,
            "n_estimators":  100,
            "random_state":  RANDOM_STATE,
            "feature_cols":  ",".join(_FEATURE_COLS),
            "n_orders":      len(labels),
        })
        mlflow.log_metrics({
            "n_anomalies":   n_anomalies,
            "anomaly_rate":  anomaly_rate,
            "score_mean":    float(scores.mean()),
            "score_std":     float(scores.std()),
            "score_min":     float(scores.min()),
            "score_max":     float(scores.max()),
        })

        # Save to disk FIRST, then log to MLflow
        model_path  = models_dir / "anomaly_model.pkl"
        scaler_path = models_dir / "anomaly_scaler.pkl"
        joblib.dump(iso,    model_path)
        joblib.dump(scaler, scaler_path)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))

        logger.info(
            "Anomaly model artifacts saved to %s | run_id=%s",
            models_dir, run.info.run_id,
        )

    return {
        "model":         iso,
        "scaler":        scaler,
        "anomaly_rate":  anomaly_rate,
        "n_anomalies":   n_anomalies,
        "feature_names": _FEATURE_COLS,
    }


def load_anomaly_model() -> dict[str, Any]:
    """Loads IsolationForest and StandardScaler from disk."""
    models_dir = ARTIFACTS_DIR / "models"
    iso:    IsolationForest = joblib.load(models_dir / "anomaly_model.pkl")
    scaler: StandardScaler  = joblib.load(models_dir / "anomaly_scaler.pkl")
    logger.info("Anomaly model loaded from %s", models_dir)
    return {"model": iso, "scaler": scaler, "feature_names": _FEATURE_COLS}


def detect_anomalies(
    df: pd.DataFrame,
    model: IsolationForest,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Adds 'is_anomaly' (bool) and 'anomaly_score' (float) columns to df.

    df must contain: final_amount_inr, mrp_inr, delivery_days, is_return (or is_returned).
    Callers must pass pre-loaded model + scaler (no silent disk-load fallback).
    """

    result = df.copy()

    # Compute discount_pct from raw fields
    result["discount_pct"] = np.clip(
        1.0 - result["final_amount_inr"] / result["mrp_inr"].replace(0, np.nan),
        0.0, 1.0,
    ).fillna(0.0)

    # Handle is_return / is_returned column name variants
    if "is_return" not in result.columns:
        result["is_return"] = result.get("is_returned", pd.Series(0, index=result.index))

    feature_df = result[_FEATURE_COLS].fillna(0.0)
    X_scaled   = scaler.transform(feature_df.values)

    labels = model.predict(X_scaled)      # -1 = anomaly, 1 = normal
    scores = model.decision_function(X_scaled)  # higher = more normal

    result["is_anomaly"]   = (labels == -1)
    result["anomaly_score"] = scores.astype(float)

    return result
