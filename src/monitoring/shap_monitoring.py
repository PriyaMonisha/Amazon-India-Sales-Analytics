"""
SHAP feature importance monitoring.

Records absolute SHAP values from /predict/churn/{id}/explain calls into a rolling
buffer. Every _EMIT_EVERY calls, computes rolling mean |SHAP| per feature and emits
Prometheus gauges for the top-10 features (ranked by baseline importance).

Drift ratio = |current_mean - baseline_mean| / baseline_mean.
Alert fires (logged as WARNING) when drift_ratio > DRIFT_ALERT_THRESHOLD (0.5).

Design decisions:
  - Top-10 only: limits Prometheus time-series count; avoids cardinality explosion.
  - Buffer cleared after each emit: prevents double-counting.
  - Feature count guard: skips update if feature set doesn't match baseline.
  - Baseline in separate file (churn_shap_baseline.json) — not mixed into prob baseline.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

from config import ARTIFACTS_DIR
from src.monitoring.metrics import SHAP_DRIFT_RATIO, SHAP_MEAN_MAGNITUDE

logger = logging.getLogger(__name__)

_TOP_N: int = 10
_EMIT_EVERY: int = 100
_DRIFT_ALERT_THRESHOLD: float = 0.50
_BUFFER_MAXLEN: int = 500

_shap_buffer: deque[dict[str, float]] = deque(maxlen=_BUFFER_MAXLEN)

_SHAP_BASELINE_PATH: Path = ARTIFACTS_DIR / "models" / "churn_shap_baseline.json"


def record_shap_values(feature_shap: dict[str, float]) -> None:
    """
    Append one explain call's absolute SHAP values to the rolling buffer.
    Called from api/routers/churn.py after every /explain response.
    """
    _shap_buffer.append({k: abs(v) for k, v in feature_shap.items()})


def maybe_update_shap_metrics() -> None:
    """
    Emit Prometheus gauges when buffer has >= _EMIT_EVERY entries.
    Clears the buffer after emitting to avoid double-counting.
    No-ops if baseline is not available yet (pre-training state).
    """
    if len(_shap_buffer) < _EMIT_EVERY:
        return

    baseline = _load_shap_baseline()
    if baseline is None:
        return

    records = list(_shap_buffer)
    _shap_buffer.clear()

    # Feature count guard — skip silently if feature set has changed since training
    if records:
        current_features = set(records[0].keys())
        baseline_features = set(baseline.get("features", []))
        if current_features != baseline_features:
            logger.warning(
                "SHAP feature mismatch: current=%d vs baseline=%d — skipping drift update",
                len(current_features), len(baseline_features),
            )
            return

    # Compute rolling mean |SHAP| per feature
    feature_means: dict[str, float] = {}
    n = len(records)
    for rec in records:
        for feat, val in rec.items():
            feature_means[feat] = feature_means.get(feat, 0.0) + val / n

    # Select top-10 by baseline importance rank (stable order, no cardinality creep)
    mean_abs_shap: dict[str, float] = baseline.get("mean_abs_shap", {})
    top_features = sorted(mean_abs_shap, key=mean_abs_shap.get, reverse=True)[:_TOP_N]  # type: ignore[arg-type]

    for feat in top_features:
        current = feature_means.get(feat)
        if current is None:
            continue
        baseline_val = mean_abs_shap.get(feat, 0.0)
        drift_ratio = abs(current - baseline_val) / max(baseline_val, 1e-6)

        SHAP_MEAN_MAGNITUDE.labels(model_name="churn", feature_name=feat).set(current)
        SHAP_DRIFT_RATIO.labels(model_name="churn", feature_name=feat).set(drift_ratio)

        if drift_ratio > _DRIFT_ALERT_THRESHOLD:
            logger.warning(
                "SHAP drift alert: feature=%s drift_ratio=%.2f (threshold=%.2f) "
                "current=%.4f baseline=%.4f",
                feat, drift_ratio, _DRIFT_ALERT_THRESHOLD, current, baseline_val,
            )


def _load_shap_baseline() -> dict[str, Any] | None:
    """Load SHAP baseline from disk. Returns None if not yet available (pre-training)."""
    if not _SHAP_BASELINE_PATH.exists():
        logger.debug("churn_shap_baseline.json not found — SHAP monitoring skipped (pre-training)")
        return None
    try:
        return json.loads(_SHAP_BASELINE_PATH.read_text())
    except Exception as exc:
        logger.error("Failed to load churn_shap_baseline.json: %s", exc)
        return None
