"""
Drift detection with Evidently 0.4.30.

Design:
  - Rolling buffer (_buffer) collects churn prediction feature dicts from the API.
  - record_churn_features(row) is called from api/routers/churn.py after every prediction.
  - maybe_run_drift_check() auto-fires (thread-safe) when buffer reaches DRIFT_CHECK_THRESHOLD.
  - flush_and_check() checks for baseline BEFORE draining buffer — if baseline missing, buffer
    stays intact for when training completes (pre-training state, Docker off).
  - compute_churn_drift(df) is the pure computation: Evidently DataDriftPreset vs baseline,
    Prometheus gauge updates, returns raw Evidently dict.

Implementation notes:
  - col_stats["drift_score"] is the KS statistic — NOT "stattest_threshold" (that is the threshold).
  - baseline_churn_proba.json["feature_distributions"] maps column_name → list of float values.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import config
from src.monitoring.metrics import (
    DRIFT_DATASET_DRIFT,
    DRIFT_DETECTED,
    DRIFT_DRIFTED_COLUMNS_SHARE,
    DRIFT_KS_STATISTIC,
)

logger = logging.getLogger(__name__)

BASELINE_PATH: Path = config.ARTIFACTS_DIR / "models" / "baseline_churn_proba.json"
_LATEST_DRIFT_PATH: Path = config.ARTIFACTS_DIR / "drift" / "latest_drift.json"
DRIFT_CHECK_THRESHOLD: int = 100      # auto-trigger after this many buffered predictions
_BUFFER_MAXLEN: int = 500             # rolling window cap; oldest records dropped on overflow
_MIN_FEATURES_FOR_RETRAIN: int = 3    # absolute count guard — avoids retraining on 1-2 feature drift

_buffer: deque[dict] = deque(maxlen=_BUFFER_MAXLEN)
_drift_lock = threading.Lock()        # prevents concurrent Evidently runs on buffer threshold


# ---------------------------------------------------------------------------
# Buffer management
# ---------------------------------------------------------------------------

def record_churn_features(row: dict) -> None:
    """Append one prediction's feature dict to the rolling buffer."""
    _buffer.append(row)


def get_drift_status() -> dict[str, Any]:
    """
    Return the latest persisted drift result without touching the in-memory buffer.

    Safe to call from Airflow DAGs (separate process — no shared buffer state).
    Returns {"dataset_drift": False, "reason": "..."} when no result is available or stale.
    A result older than 24 hours is treated as stale.
    """
    if not _LATEST_DRIFT_PATH.exists():
        return {"dataset_drift": False, "reason": "no_check_run_yet"}
    try:
        data: dict = json.loads(_LATEST_DRIFT_PATH.read_text())
        checked_at = datetime.fromisoformat(data["checked_at"])
        age_s = (datetime.utcnow() - checked_at).total_seconds()
        if age_s > 86400:
            return {"dataset_drift": False, "reason": "stale", "age_seconds": age_s}
        return data
    except Exception as exc:
        logger.error("Failed to read latest_drift.json: %s", exc)
        return {"dataset_drift": False, "reason": "read_error"}


def maybe_run_drift_check() -> None:
    """Auto-trigger one drift check (thread-safe) when buffer >= DRIFT_CHECK_THRESHOLD."""
    if len(_buffer) >= DRIFT_CHECK_THRESHOLD:
        if _drift_lock.acquire(blocking=False):
            try:
                flush_and_check()
            finally:
                _drift_lock.release()


def flush_and_check() -> dict[str, Any]:
    """
    Run drift check against current buffer contents.

    Baseline is checked FIRST — if missing, buffer is left intact so records are not lost
    before model training completes (pre-training / Docker-off state).

    Returns raw Evidently result dict, or {} if skipped.
    """
    ref_df = _load_reference_df()   # load once — pass through to avoid double JSON read
    if ref_df is None:
        # Do NOT drain buffer — keep records for when baseline becomes available
        return {}
    records = list(_buffer)
    _buffer.clear()
    if not records:
        return {}
    return compute_churn_drift(pd.DataFrame(records), reference_df=ref_df)


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def _load_reference_df() -> pd.DataFrame | None:
    """
    Load baseline feature distributions from baseline_churn_proba.json.

    Returns None if file missing (pre-training state) — callers must handle this gracefully.
    Key structure: data["feature_distributions"] maps column_name → list of float values.
    """
    if not BASELINE_PATH.exists():
        logger.debug("baseline_churn_proba.json not found — drift check skipped (pre-training)")
        return None
    try:
        data: dict = json.loads(BASELINE_PATH.read_text())
        feat_dist: dict[str, list] = data.get("feature_distributions", {})
        if not feat_dist:
            logger.warning("baseline_churn_proba.json missing 'feature_distributions' key")
            return None
        return pd.DataFrame(feat_dist)
    except Exception as exc:
        logger.error("Failed to load baseline JSON: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core drift computation
# ---------------------------------------------------------------------------

def compute_churn_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Run Evidently DataDriftPreset on current_df vs baseline reference.

    Pass reference_df from flush_and_check() to avoid loading JSON twice per check.
    If reference_df is None (standalone/test call), it is loaded from disk.

    Updates Prometheus gauges (all imported from src.monitoring.metrics):
        DRIFT_KS_STATISTIC          — per-feature KS score
        DRIFT_DETECTED              — per-feature drift flag (0 | 1)
        DRIFT_DATASET_DRIFT         — overall dataset drift flag (0 | 1)
        DRIFT_DRIFTED_COLUMNS_SHARE — fraction of features drifted

    Returns raw Evidently report.as_dict() — the /monitor/drift/run endpoint parses this.
    Returns {} on any failure so callers always get a dict.
    """
    from evidently.metric_preset import DataDriftPreset  # lazy import (heavy startup cost)
    from evidently.report import Report

    if reference_df is None:
        reference_df = _load_reference_df()
    if reference_df is None or current_df.empty:
        return {}

    # Align schemas — Evidently requires matching columns
    common_cols = [c for c in reference_df.columns if c in current_df.columns]
    if not common_cols:
        logger.warning(
            "No common columns between reference (%s) and current (%s)",
            list(reference_df.columns), list(current_df.columns),
        )
        return {}

    ref = reference_df[common_cols].reset_index(drop=True)
    cur = current_df[common_cols].reset_index(drop=True)

    report = Report(metrics=[DataDriftPreset()])
    try:
        report.run(reference_data=ref, current_data=cur)
    except Exception as exc:
        logger.error("Evidently report.run() failed: %s", exc)
        return {}

    result: dict = report.as_dict()

    # --- Parse Evidently 0.4.30 result structure (Rule 9) ---
    try:
        top_result: dict        = result["metrics"][0]["result"]
        drift_by_cols: dict     = top_result["drift_by_columns"]   # col_name = dict key
        dataset_drift: bool     = top_result.get("dataset_drift", False)
        n_total: int            = top_result.get("number_of_columns", len(common_cols))
        n_drifted: int          = top_result.get("number_of_drifted_columns", 0)
    except (KeyError, IndexError) as exc:
        logger.error(
            "Unexpected Evidently result structure: %s\nTop-level keys: %s",
            exc, list(result.keys()),
        )
        return result

    # --- Update Prometheus gauges ---
    DRIFT_DATASET_DRIFT.labels(model_name="churn").set(1 if dataset_drift else 0)
    share = n_drifted / n_total if n_total > 0 else 0.0
    DRIFT_DRIFTED_COLUMNS_SHARE.labels(model_name="churn").set(share)

    for col_name, col_stats in drift_by_cols.items():
        ks_stat: float  = col_stats.get("drift_score", 0.0)          # Rule 9: drift_score key
        detected: int   = 1 if col_stats.get("drift_detected", False) else 0
        DRIFT_KS_STATISTIC.labels(model_name="churn", feature_name=col_name).set(ks_stat)
        DRIFT_DETECTED.labels(model_name="churn", feature_name=col_name).set(detected)
        logger.info("drift[churn][%s]: ks=%.4f detected=%s", col_name, ks_stat, bool(detected))

    logger.info(
        "Drift check complete — %d/%d features drifted (%.0f%%), dataset_drift=%s",
        n_drifted, n_total, share * 100, dataset_drift,
    )

    # Persist summary to disk so Airflow DAGs can read it (separate process — no shared buffer)
    drift_summary = {
        "dataset_drift": dataset_drift,
        "n_drifted":     n_drifted,
        "n_total":       n_total,
        "share":         share,
        "checked_at":    datetime.utcnow().isoformat(),
    }
    try:
        _LATEST_DRIFT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LATEST_DRIFT_PATH.write_text(json.dumps(drift_summary, indent=2))
    except Exception as exc:
        logger.warning("Could not persist latest_drift.json: %s", exc)

    return result
