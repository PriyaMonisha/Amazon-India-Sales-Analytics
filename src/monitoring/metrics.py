"""
Prometheus metric registry — single source of truth for all metric names and label names.

Label names must be IDENTICAL across this file, drift.py, shap_monitoring.py, and Grafana JSON:
    model_name   — one of MODEL_NAMES: churn | forecast | pricing | recommend | anomaly
    feature_name — individual feature column name (churn numeric features)

Import from here everywhere. Never re-define or re-name metrics in other modules.
Mismatched labels produce empty query results in Grafana without any error.
"""
from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Single source of truth for Prometheus label values
# Used in: lifespan MODEL_LOADED gauges, middleware regex, Grafana panel labels
# ---------------------------------------------------------------------------
MODEL_NAMES: list[str] = ["churn", "forecast", "pricing", "recommend", "anomaly"]

# ---------------------------------------------------------------------------
# Request tracking
# ---------------------------------------------------------------------------
PREDICTION_REQUEST_COUNTER = Counter(
    "amazon_prediction_requests_total",
    "Total prediction requests by model and outcome",
    ["model_name", "status"],   # status: success | error
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "amazon_prediction_latency_seconds",
    "End-to-end prediction request latency in seconds",
    ["model_name"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ---------------------------------------------------------------------------
# Model health
# ---------------------------------------------------------------------------
MODEL_LOADED = Gauge(
    "amazon_model_loaded",
    "1 if the model artifact is loaded and ready for inference",
    ["model_name"],
)

# ---------------------------------------------------------------------------
# Drift detection
# Label names are IDENTICAL in drift.py (gauge .labels() calls) and Grafana JSON queries
# ---------------------------------------------------------------------------
DRIFT_KS_STATISTIC = Gauge(
    "amazon_drift_ks_statistic",
    "Kolmogorov-Smirnov drift score from Evidently (per feature)",
    ["model_name", "feature_name"],
)

DRIFT_DETECTED = Gauge(
    "amazon_drift_detected",
    "1 if Evidently flagged drift for this feature, 0 otherwise",
    ["model_name", "feature_name"],
)

DRIFT_DATASET_DRIFT = Gauge(
    "amazon_drift_dataset_drift",
    "1 if Evidently flagged overall dataset drift for this model",
    ["model_name"],
)

DRIFT_DRIFTED_COLUMNS_SHARE = Gauge(
    "amazon_drift_drifted_columns_share",
    "Fraction of features with drift detected (0.0 to 1.0)",
    ["model_name"],
)

# ---------------------------------------------------------------------------
# Churn business metrics
# 0.95 and 1.0 added so highest-risk customers (0.9–1.0 range) are visible in Grafana
# ---------------------------------------------------------------------------
CHURN_PROBABILITY_HISTOGRAM = Histogram(
    "amazon_churn_probability",
    "Distribution of predicted churn probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# ---------------------------------------------------------------------------
# SHAP feature importance monitoring
# Only top-10 features emitted per check to control Prometheus cardinality.
# Label names IDENTICAL to drift metrics (model_name, feature_name).
# ---------------------------------------------------------------------------
SHAP_MEAN_MAGNITUDE = Gauge(
    "amazon_shap_mean_magnitude",
    "Rolling mean absolute SHAP value for top-10 features (updated every 100 explain calls)",
    ["model_name", "feature_name"],
)

SHAP_DRIFT_RATIO = Gauge(
    "amazon_shap_drift_ratio",
    "Relative drift of SHAP magnitude vs training baseline: |current-baseline|/baseline",
    ["model_name", "feature_name"],
)
