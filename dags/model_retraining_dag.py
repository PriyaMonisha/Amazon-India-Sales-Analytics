"""
Drift-triggered churn model retraining DAG.

Schedule: weekly (@weekly) + on-demand manual trigger.

Pipeline:
    check_drift → retrain_churn → validate_quality → validate_artifacts → promote_model

Drift sensor:
    Reads artifacts/drift/latest_drift.json (written by the FastAPI drift endpoint).
    Uses file-based state so Airflow (separate process) doesn't share API memory.
    Short-circuits if:
        - No drift result exists yet (pre-training state)
        - Result is stale (> 24 hours old)
        - dataset_drift=False
        - share < DRIFT_RETRAIN_THRESHOLD (0.30) — avoids retraining on noise
        - n_drifted < MIN_FEATURES (3) — absolute count guard

Promotion:
    New model registered as "candidate" by train_churn_model().
    promote_model task calls registry.promote() only after quality + artifact validation.
    Keeps old version as "retired" for rollback via registry.rollback("churn").
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator

# Make project root importable inside Airflow workers
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_DRIFT_RETRAIN_THRESHOLD: float = 0.30   # mirror of config.DRIFT_RETRAIN_THRESHOLD
_MIN_FEATURES: int = 3                   # absolute guard against tiny-sample drift

_REQUIRED_ARTIFACTS: list[str] = [
    "churn_model.json",
    "churn_explainer.json",
    "churn_encoder.joblib",
    "churn_threshold.json",
    "churn_quality.json",
    "metadata.json",
]


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def _check_drift() -> bool:
    """
    ShortCircuitOperator callable.
    Returns True (proceed) only when drift is real, fresh, and above threshold.
    """
    from src.monitoring.drift import get_drift_status
    status = get_drift_status()
    if status.get("reason"):
        logger.info("Drift check skipped: %s", status["reason"])
        return False
    if not status.get("dataset_drift", False):
        logger.info("No dataset drift detected — skipping retraining")
        return False
    share    = float(status.get("share", 0))
    n_drifted = int(status.get("n_drifted", 0))
    if share < _DRIFT_RETRAIN_THRESHOLD:
        logger.info("Drift share=%.2f below threshold %.2f — skipping", share, _DRIFT_RETRAIN_THRESHOLD)
        return False
    if n_drifted < _MIN_FEATURES:
        logger.info("Only %d features drifted (min=%d) — skipping", n_drifted, _MIN_FEATURES)
        return False
    logger.info("Drift triggered retraining: share=%.2f, n_drifted=%d", share, n_drifted)
    return True


def _validate_quality(**context: object) -> str:
    """
    Validates the most recent candidate version's quality gate.
    Returns the version string (pushed to XCom for downstream tasks).
    """
    from config import ARTIFACTS_DIR
    from src.model_registry.registry import list_versions
    versions = list_versions("churn")
    candidates = [v for v in versions if v["status"] == "candidate"]
    if not candidates:
        raise RuntimeError("No candidate version found after training — check training logs")
    latest = candidates[0]
    q_path = ARTIFACTS_DIR / latest["artifact_dir"] / "churn_quality.json"
    q = json.loads(q_path.read_text())
    if not q.get("meets_threshold", False):
        raise RuntimeError(
            f"New churn model ROC-AUC={q.get('roc_auc', 'N/A')} is below production threshold. "
            "Not promoting. Investigate training data or hyperparameters."
        )
    logger.info("Quality gate passed: ROC-AUC=%.4f", q["roc_auc"])
    ti = context["ti"]
    ti.xcom_push(key="version", value=latest["version"])
    return latest["version"]


def _validate_artifacts(**context: object) -> None:
    """
    Verifies all required artifact files exist + encoder checksum before promoting.
    """
    from config import ARTIFACTS_DIR
    from src.model_registry.registry import list_versions
    ti = context["ti"]
    version: str = ti.xcom_pull(task_ids="validate_quality", key="version")
    records = list_versions("churn")
    record = next((r for r in records if r["version"] == version), None)
    if record is None:
        raise RuntimeError(f"Version '{version}' not found in registry")
    vdir = ARTIFACTS_DIR / record["artifact_dir"]
    for fname in _REQUIRED_ARTIFACTS:
        if not (vdir / fname).exists():
            raise FileNotFoundError(f"Missing required artifact: {vdir / fname}")
    sha_path = vdir / "churn_encoder.sha256"
    if sha_path.exists():
        expected = sha_path.read_text().strip()
        actual   = hashlib.sha256((vdir / "churn_encoder.joblib").read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Encoder checksum mismatch for version '{version}' — possible corruption"
            )
    logger.info("Artifact validation passed for version '%s'", version)
    ti.xcom_push(key="version", value=version)


def _promote_model(**context: object) -> None:
    """Promotes validated candidate to production in the model registry."""
    from src.model_registry.registry import promote
    ti = context["ti"]
    version: str = ti.xcom_pull(task_ids="validate_artifacts", key="version")
    promote("churn", version)
    logger.info("Promoted churn model version '%s' to production", version)


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="model_retraining",
    description="Drift-triggered churn model retraining and promotion",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "churn", "retraining"],
) as dag:

    check_drift = ShortCircuitOperator(
        task_id="check_drift",
        python_callable=_check_drift,
    )

    retrain_churn = BashOperator(
        task_id="retrain_churn",
        bash_command=f"python {_PROJECT_ROOT}/scripts/train_churn.py",
    )

    validate_quality = PythonOperator(
        task_id="validate_quality",
        python_callable=_validate_quality,
        provide_context=True,
    )

    validate_artifacts = PythonOperator(
        task_id="validate_artifacts",
        python_callable=_validate_artifacts,
        provide_context=True,
    )

    promote_model = PythonOperator(
        task_id="promote_model",
        python_callable=_promote_model,
        provide_context=True,
    )

    check_drift >> retrain_churn >> validate_quality >> validate_artifacts >> promote_model
