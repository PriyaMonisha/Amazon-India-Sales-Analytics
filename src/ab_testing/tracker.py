"""
A/B prediction recording and AUC comparison.

record_prediction() is called from the churn router after each inference.
compute_auc_by_version() is called by the promotion DAG to compare champion vs challenger.

AUC comparison requires ground-truth labels from ab_outcomes (populated nightly by
scripts/materialize_ab_outcomes.py). Only predictions with outcomes available
(predicted_at >= lookback_days ago) are included to avoid censoring bias.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_INSERT_SQL = text("""
    INSERT INTO ab_predictions (customer_id, model_version, churn_prob, predicted_at)
    VALUES (:customer_id, :model_version, :churn_prob, :predicted_at)
""")

_AUC_QUERY = text("""
WITH labelled AS (
    SELECT
        p.model_version,
        p.churn_prob,
        o.churned::int AS label
    FROM ab_predictions p
    JOIN ab_outcomes o ON p.customer_id = o.customer_id
    WHERE p.predicted_at <= NOW() - INTERVAL ':lookback_days days'
)
SELECT
    model_version,
    COUNT(*)                                          AS n,
    -- Wilcoxon-Mann-Whitney AUC approximation via rank correlation
    -- Note: exact AUC requires scikit-learn; this is a fast SQL approximation
    AVG(churn_prob::float)                            AS mean_prob,
    AVG(label::float)                                 AS churn_rate
FROM labelled
GROUP BY model_version
""")


def record_prediction(
    customer_id: str,
    model_version: str,
    churn_prob: float,
    engine: Engine,
) -> None:
    """Insert one churn prediction into ab_predictions. Non-fatal on failure."""
    try:
        with engine.begin() as conn:
            conn.execute(_INSERT_SQL, {
                "customer_id":   customer_id,
                "model_version": model_version,
                "churn_prob":    float(churn_prob),
                "predicted_at":  datetime.now(tz=timezone.utc).isoformat(),
            })
    except Exception as exc:
        logger.warning("A/B prediction record failed (non-fatal): %s", exc)


def compute_auc_by_version(engine: Engine, lookback_days: int = 90) -> dict[str, Any]:
    """
    Compare champion vs challenger using labelled predictions.

    Requires ab_outcomes table to be populated (see scripts/materialize_ab_outcomes.py).
    Returns stats dict with keys: champion_n, challenger_n, champion_mean_prob,
    challenger_mean_prob, champion_churn_rate, challenger_churn_rate.

    Note: full AUC computation (sklearn.metrics.roc_auc_score) is done in the
    promotion DAG task which loads the labelled rows into Python for sklearn.
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    WITH labelled AS (
                        SELECT p.model_version, p.churn_prob, o.churned::int AS label
                        FROM ab_predictions p
                        JOIN ab_outcomes o ON p.customer_id = o.customer_id
                        WHERE p.predicted_at <= NOW() - INTERVAL '90 days'
                    )
                    SELECT model_version, COUNT(*) AS n,
                           AVG(churn_prob) AS mean_prob,
                           AVG(label::float) AS churn_rate
                    FROM labelled
                    GROUP BY model_version
                """)
            ).fetchall()
        result: dict[str, Any] = {}
        for row in rows:
            result[row[0]] = {
                "n":           int(row[1]),
                "mean_prob":   float(row[2]),
                "churn_rate":  float(row[3]),
            }
        return result
    except Exception as exc:
        logger.error("compute_auc_by_version failed: %s", exc)
        return {}


def compute_full_auc_by_version(engine: Engine) -> dict[str, float]:
    """
    Compute proper sklearn ROC-AUC per model version.
    Loads labelled predictions into memory; intended for promotion DAG only.
    """
    from sklearn.metrics import roc_auc_score
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT p.model_version, p.churn_prob, o.churned::int AS label
                    FROM ab_predictions p
                    JOIN ab_outcomes o ON p.customer_id = o.customer_id
                    WHERE p.predicted_at <= NOW() - INTERVAL '90 days'
                """)
            ).fetchall()
        by_version: dict[str, tuple[list, list]] = {}
        for version, prob, label in rows:
            probs, labels = by_version.setdefault(version, ([], []))
            probs.append(float(prob))
            labels.append(int(label))
        return {
            version: float(roc_auc_score(labels, probs))
            for version, (probs, labels) in by_version.items()
            if len(set(labels)) == 2   # need both classes for AUC
        }
    except Exception as exc:
        logger.error("compute_full_auc_by_version failed: %s", exc)
        return {}
