"""
A/B champion vs challenger promotion DAG.

Schedule: @weekly.

Pipeline:
    materialize_outcomes → compute_auc → check_criteria → promote_challenger

Promotion criterion:
    challenger ROC-AUC >= champion ROC-AUC + 0.01
    AND n_challenger >= MIN_SAMPLES_FOR_PROMOTION (1000)

After promotion:
    - challenger version becomes "production" in the registry
    - ab_config.json is reset (challenger_version=null, enabled=false)
    - The API will serve the promoted model on next startup / hot-reload

Rollback (manual):
    python -c "from src.model_registry.registry import rollback; rollback('churn')"
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_MIN_SAMPLES: int = 1000
_AUC_MARGIN: float = 0.01           # challenger must beat champion by this margin
_AB_CONFIG_PATH = _PROJECT_ROOT / "artifacts" / "ab_config.json"


def _compute_and_check(**context: object) -> bool:
    """
    ShortCircuitOperator callable.
    Loads labelled predictions, computes AUC per version, checks promotion criteria.
    Pushes auc dict to XCom and returns True if promotion should proceed.
    """
    from config import DB_URL
    from sqlalchemy import create_engine
    from src.ab_testing.tracker import compute_full_auc_by_version, compute_auc_by_version

    engine = create_engine(os.environ.get("AMAZON_DB_URL", DB_URL))
    counts = compute_auc_by_version(engine)
    n_challenger = counts.get("challenger", {}).get("n", 0)

    if n_challenger < _MIN_SAMPLES:
        logger.info(
            "Insufficient challenger samples: %d < %d — skipping promotion",
            n_challenger, _MIN_SAMPLES,
        )
        return False

    auc_by_version = compute_full_auc_by_version(engine)
    champion_auc   = auc_by_version.get("champion", 0.0)
    challenger_auc = auc_by_version.get("challenger", 0.0)

    logger.info(
        "A/B AUC: champion=%.4f challenger=%.4f (need challenger >= champion + %.2f)",
        champion_auc, challenger_auc, _AUC_MARGIN,
    )

    if challenger_auc < champion_auc + _AUC_MARGIN:
        logger.info("Challenger does not beat champion — no promotion")
        return False

    context["ti"].xcom_push(key="auc_by_version", value=auc_by_version)
    return True


def _promote_challenger(**context: object) -> None:
    """
    Promotes the challenger version to production and resets ab_config.json.
    """
    from src.model_registry.registry import promote

    # Read challenger version from ab_config
    if not _AB_CONFIG_PATH.exists():
        raise RuntimeError("ab_config.json not found — cannot identify challenger version")
    ab_cfg: dict = json.loads(_AB_CONFIG_PATH.read_text())
    challenger_version = ab_cfg.get("challenger_version")
    if not challenger_version:
        raise RuntimeError("ab_config.json has no challenger_version set")

    auc_by_version: dict = context["ti"].xcom_pull(task_ids="compute_and_check", key="auc_by_version")
    logger.info(
        "Promoting challenger version '%s' (AUC=%.4f) over champion (AUC=%.4f)",
        challenger_version,
        auc_by_version.get("challenger", 0.0),
        auc_by_version.get("champion", 0.0),
    )

    promote("churn", challenger_version)

    # Reset A/B config — next experiment starts with a fresh challenger
    ab_cfg["challenger_version"] = None
    ab_cfg["enabled"] = False
    _AB_CONFIG_PATH.write_text(json.dumps(ab_cfg, indent=2))
    logger.info("A/B config reset — experiment concluded")


with DAG(
    dag_id="ab_promotion",
    description="Weekly A/B champion vs challenger promotion check",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "ab_testing", "churn"],
) as dag:

    materialize_outcomes = BashOperator(
        task_id="materialize_outcomes",
        bash_command=f"python {_PROJECT_ROOT}/scripts/materialize_ab_outcomes.py",
    )

    compute_and_check = ShortCircuitOperator(
        task_id="compute_and_check",
        python_callable=_compute_and_check,
        provide_context=True,
    )

    promote_challenger = PythonOperator(
        task_id="promote_challenger",
        python_callable=_promote_challenger,
        provide_context=True,
    )

    materialize_outcomes >> compute_and_check >> promote_challenger
