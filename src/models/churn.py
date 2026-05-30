import hashlib
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing_extensions import TypedDict

import config
from config import (
    ARTIFACTS_DIR,
    CV_FOLDS,
    N_OPTUNA_TRIALS,
    RANDOM_STATE,
    SAMPLE_ROWS,
)
from src.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

# --- Constants ---
TRAIN_REF_DATE    = date(2023, 1, 1)   # features < 2023-01-01; label Jan–Mar 2023
TEST_REF_DATE     = date(2024, 1, 1)   # features < 2024-01-01; label Jan–Mar 2024
CHURN_WINDOW_DAYS = 90
DATASET_END_DATE  = date(2025, 12, 31)
BASELINE_SAMPLE_SIZE = 2000  # rows sampled from test set for Evidently reference JSON

CATEGORICAL_FEATURES: list[str] = ["rfm_segment", "preferred_category"]
NUMERIC_FEATURES: list[str] = [
    "days_since_last_purchase",
    "total_orders_90d",
    "total_orders_all_time",
    "avg_order_value_last_6m",
    "avg_order_value_all_time",
    "total_spend_all_time",
    "is_prime_member",
    "recency_score",
    "frequency_score",
    "monetary_score",
    "unique_categories_purchased",
    "return_rate_historical",
]
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # order = training contract


# --- TypedDict contract with FastAPI (Section 5) ---
class ChurnModelBundle(TypedDict):
    model: xgb.XGBClassifier
    explainer: Any               # shap.TreeExplainer
    encoder: OrdinalEncoder
    feature_names: list[str]     # ALL_FEATURES in exact training order
    categorical_features: list[str]
    threshold: float
    churn_rate_train: float
    churn_rate_test: float


# --- SQL ---
_CHURN_QUERY = text("""
WITH features AS (
    SELECT
        ft.customer_id,
        (CAST(:ref_date AS date) - MAX(ft.order_date)::date) AS days_since_last_purchase,
        COUNT(*) FILTER (
            WHERE ft.order_date >= CAST(:ref_date AS date) - INTERVAL '90 days'
        )                                                  AS total_orders_90d,
        COUNT(*)                                           AS total_orders_all_time,
        AVG(ft.final_amount_inr) FILTER (
            WHERE ft.order_date >= CAST(:ref_date AS date) - INTERVAL '180 days'
        )                                                  AS avg_order_value_last_6m,
        AVG(ft.final_amount_inr)                           AS avg_order_value_all_time,
        SUM(ft.final_amount_inr)                           AS total_spend_all_time,
        COUNT(DISTINCT dp.subcategory)                     AS unique_categories_purchased,
        COALESCE(
            MODE() WITHIN GROUP (ORDER BY dp.category), 'Unknown'
        )                                                  AS preferred_category,
        AVG(
            CASE WHEN ft.return_status = 'Returned' THEN 1.0 ELSE 0.0 END
        )                                                  AS return_rate_historical
    FROM fact_transactions ft
    LEFT JOIN dim_products dp ON ft.product_id = dp.product_id
    WHERE ft.order_date < CAST(:ref_date AS date)
    GROUP BY ft.customer_id
),
label_window AS (
    SELECT customer_id
    FROM fact_transactions
    WHERE order_date >= CAST(:ref_date AS date)
      AND order_date < CAST(:obs_end AS date)
    GROUP BY customer_id
)
SELECT
    f.*,
    dc.is_prime_member::int                                AS is_prime_member,
    CASE WHEN l.customer_id IS NULL THEN 1 ELSE 0 END     AS churned
FROM features f
JOIN  dim_customers dc ON f.customer_id = dc.customer_id
LEFT JOIN label_window l  ON f.customer_id = l.customer_id
""")


# --- RFM helpers (same logic as compute.py — duplicated to avoid importing private funcs) ---

def _compute_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    df = df.copy()
    df["recency_score"] = pd.cut(
        df["days_since_last_purchase"].rank(pct=True, ascending=False),
        bins=bins, labels=[5, 4, 3, 2, 1], include_lowest=True,
    ).astype(int)
    df["frequency_score"] = pd.cut(
        df["total_orders_all_time"].rank(pct=True, ascending=True),
        bins=bins, labels=[1, 2, 3, 4, 5], include_lowest=True,
    ).astype(int)
    df["monetary_score"] = pd.cut(
        df["total_spend_all_time"].rank(pct=True, ascending=True),
        bins=bins, labels=[1, 2, 3, 4, 5], include_lowest=True,
    ).astype(int)
    return df


def _assign_rfm_segment(df: pd.DataFrame) -> pd.Series:
    r, f, m = df["recency_score"], df["frequency_score"], df["monetary_score"]
    # Priority: first matching condition wins (np.select behaviour).
    # Champion before Loyal; New before At Risk for high-recency, low-frequency customers.
    conditions = [
        (r >= 4) & (f >= 4) & (m >= 4),
        (r >= 3) & (f >= 3) & (m >= 3),
        (r >= 4) & (f <= 2),
        (r <= 2) & (f >= 3) & (m >= 3),
        (r <= 2) & (f <= 2) & (m <= 2),
        (m >= 4) & (f <= 2),
    ]
    choices = ["Champion", "Loyal", "New", "At Risk", "Lost", "High Value Occasional"]
    return pd.Series(np.select(conditions, choices, default="Potential"), index=df.index)


# --- Data loader ---

def _load_churn_data(engine: Engine, ref_date: date) -> pd.DataFrame:
    """Loads features + churn label for all customers observed at ref_date."""
    obs_end = ref_date + timedelta(days=CHURN_WINDOW_DAYS)
    if obs_end > DATASET_END_DATE:
        raise ValueError(
            f"ref_date={ref_date} + {CHURN_WINDOW_DAYS}d = {obs_end} "
            f"exceeds dataset end {DATASET_END_DATE}"
        )

    logger.info("Loading churn data: ref_date=%s, obs_end=%s", ref_date, obs_end)
    with engine.connect() as conn:
        df = pd.read_sql(
            _CHURN_QUERY,
            conn,
            params={"ref_date": str(ref_date), "obs_end": str(obs_end)},
        )

    if df.empty:
        logger.warning("No churn data returned for ref_date=%s", ref_date)
        return df

    # Post-SQL null guards
    df["is_prime_member"] = df["is_prime_member"].fillna(0).astype(int)
    df["avg_order_value_last_6m"] = df["avg_order_value_last_6m"].fillna(
        df["avg_order_value_all_time"]
    )
    df["total_orders_90d"] = df["total_orders_90d"].fillna(0).astype(int)

    # RFM
    df = _compute_rfm_scores(df)
    df["rfm_segment"] = _assign_rfm_segment(df)

    churn_rate = df["churned"].mean()
    logger.info(
        "Loaded %d customers for ref_date=%s | churn_rate=%.1f%%",
        len(df), ref_date, churn_rate * 100,
    )
    return df


# --- Threshold optimisation ---

def _find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """F1-maximising threshold from precision-recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    denom = precisions + recalls + 1e-10
    f1_scores = np.where(denom > 0, 2 * precisions * recalls / denom, 0.0)
    # thresholds has len = len(precisions) - 1
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx])


# --- JSON encoder for numpy types ---

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# --- Public API ---

def train_churn_model(engine: Engine) -> ChurnModelBundle:
    """
    Trains XGBoost churn classifier with MLflow tracking.

    Training set: customer features as of 2023-01-01; label = no purchase Jan–Mar 2023.
    Test set:     customer features as of 2024-01-01; label = no purchase Jan–Mar 2024.
    Temporal split prevents future-data leakage. Same customers may appear in both sets.

    Artifacts written to artifacts/models/:
        churn_model.json, churn_explainer.pkl, churn_encoder.pkl,
        churn_feature_names.json, baseline_churn_proba.json
    """
    models_dir = ARTIFACTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Step 1–3: load data (right-censoring guard is inside _load_churn_data)
    df_train = _load_churn_data(engine, TRAIN_REF_DATE)
    df_test  = _load_churn_data(engine, TEST_REF_DATE)

    if df_train.empty or df_test.empty:
        raise RuntimeError(
            "Churn data load returned empty DataFrame — run ETL pipeline first"
        )

    # Step 4: churn rate sanity check (relaxed in FAST_MODE — sparse sample skews rates)
    train_churn_rate = float(df_train["churned"].mean())
    _lo, _hi = (0.01, 0.999) if config.FAST_MODE else (0.05, 0.70)
    if not (_lo < train_churn_rate < _hi):
        raise ValueError(
            f"Churn rate {train_churn_rate:.1%} outside expected range [{_lo:.0%}, {_hi:.0%}]. "
            "Check label_window SQL — wrong obs_end, INNER JOIN dropping churned customers, "
            "or ref_date outside dataset range."
        )
    if train_churn_rate > 0.70:
        logger.warning(
            "High churn rate %.1f%% — expected with FAST_MODE sparse sample. "
            "Run with FAST_MODE=false for representative rates.", train_churn_rate * 100
        )

    # Step 5: sort by customer_id so StratifiedKFold fold assignment is deterministic
    # without seeding KFold — shuffle=False means fold order = row order.
    df_train = df_train.sort_values("customer_id").reset_index(drop=True)

    # Step 6–7: fit OrdinalEncoder on train ONLY, transform both
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_train = df_train.copy()
    df_test  = df_test.copy()
    df_train[CATEGORICAL_FEATURES] = encoder.fit_transform(
        df_train[CATEGORICAL_FEATURES]
    )
    df_test[CATEGORICAL_FEATURES] = encoder.transform(df_test[CATEGORICAL_FEATURES])

    # Step 8: FAST_MODE sample AFTER encoding (not before)
    if SAMPLE_ROWS is not None:
        df_train = df_train.sample(
            n=min(SAMPLE_ROWS, len(df_train)), random_state=RANDOM_STATE
        )
        logger.info("FAST_MODE: sampled %d train rows", len(df_train))

    # Step 9: derive X/y AFTER sampling — avoids stale array bug
    X_train = df_train[ALL_FEATURES].values
    y_train = df_train["churned"].values
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    # Step 10: test arrays (never sampled — test is always full)
    X_test    = df_test[ALL_FEATURES].values
    y_test    = df_test["churned"].values
    X_test_df = df_test[ALL_FEATURES]  # named DataFrame for baseline JSON

    logger.info(
        "Train: %d rows, churn=%.1f%% | Test: %d rows, churn=%.1f%% | scale_pos_weight=%.2f",
        len(y_train), y_train.mean() * 100,
        len(y_test),  y_test.mean() * 100,
        scale_pos_weight,
    )

    # Step 11–12: MLflow
    setup_mlflow("amazon_churn")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with mlflow.start_run(run_name="churn_xgboost") as run:

        # Step 13: Optuna hyperparameter search — CV on train ONLY
        def _objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
                "max_depth":         trial.suggest_int("max_depth", 3, 8),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 5.0),
                "scale_pos_weight":  scale_pos_weight,
                "random_state":      RANDOM_STATE,
                "objective":         "binary:logistic",
                "eval_metric":       "auc",
                "tree_method":       "hist",
            }
            clf = xgb.XGBClassifier(**params)
            cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=False)
            fold_scores = []
            for tr_idx, va_idx in cv.split(X_train, y_train):
                X_tr = X_train[tr_idx]
                y_tr = y_train[tr_idx]
                X_va = X_train[va_idx]
                y_va = y_train[va_idx]
                clf.fit(X_tr, y_tr, verbose=False)
                proba = clf.predict_proba(X_va)[:, 1]
                fold_scores.append(float(roc_auc_score(y_va, proba)))
            return float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

        best_params: dict[str, Any] = dict(study.best_params)
        best_params.update({
            "scale_pos_weight": scale_pos_weight,
            "random_state":     RANDOM_STATE,
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "tree_method":      "hist",
        })
        mlflow.log_params(best_params)
        logger.info("Best Optuna ROC-AUC (CV): %.4f", study.best_value)

        # Step 14: train final model on full (possibly sampled) train set
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train, verbose=False)

        # Step 15–16: evaluate on 2024 test set
        y_proba_test = model.predict_proba(X_test)[:, 1]
        optimal_threshold = _find_optimal_threshold(y_test, y_proba_test)
        y_pred_test = (y_proba_test >= optimal_threshold).astype(int)

        # Step 17: metrics
        roc_auc  = float(roc_auc_score(y_test, y_proba_test))
        avg_prec = float(average_precision_score(y_test, y_proba_test))
        report   = classification_report(y_test, y_pred_test, output_dict=True)

        if roc_auc < config.CHURN_MIN_ROC_AUC:
            logger.warning(
                "ROC-AUC %.4f is below production threshold %.4f",
                roc_auc, config.CHURN_MIN_ROC_AUC,
            )

        metrics = {
            "roc_auc":          roc_auc,
            "avg_precision":    avg_prec,
            "precision_churn":  report["1"]["precision"],
            "recall_churn":     report["1"]["recall"],
            "f1_churn":         report["1"]["f1-score"],
            "optimal_threshold": optimal_threshold,
            "train_samples":    int(len(y_train)),
            "test_samples":     int(len(y_test)),
            "churn_rate_train": float(y_train.mean()),
            "churn_rate_test":  float(y_test.mean()),
        }
        mlflow.log_metrics(metrics)
        logger.info(
            "Churn model — ROC-AUC=%.4f | Avg-Prec=%.4f | F1=%.4f | threshold=%.3f",
            roc_auc, avg_prec, report["1"]["f1-score"], optimal_threshold,
        )

        # Step 18: save artifacts to disk FIRST, then log to MLflow

        # 18a-i: quality gate (load_churn_model refuses to load if meets_threshold=False)
        quality = {
            "roc_auc":         float(roc_auc),
            "f1_churn":        float(report["1"]["f1-score"]),
            "threshold":       float(optimal_threshold),
            "meets_threshold": roc_auc >= config.CHURN_MIN_ROC_AUC,
        }
        (models_dir / "churn_quality.json").write_text(json.dumps(quality, indent=2))
        mlflow.log_artifact(str(models_dir / "churn_quality.json"))

        # 18a-ii: XGBoost model (native JSON — not pickle)
        model_path = models_dir / "churn_model.json"
        model.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))

        # 18b: SHAP TreeExplainer (JSON — no pickle, no arbitrary code execution on load)
        explainer = shap.TreeExplainer(model)
        explainer_path = models_dir / "churn_explainer.json"
        explainer.save(str(explainer_path))
        mlflow.log_artifact(str(explainer_path))

        # 18b-ii: SHAP baseline mean |SHAP| per feature (used by shap_monitoring.py for drift detection)
        _shap_sample_n = min(500, len(X_test))
        _sv = explainer.shap_values(X_test[:_shap_sample_n])
        _shap_arr = _sv[1] if isinstance(_sv, list) else _sv   # positive class for classifiers
        _mean_abs_shap = dict(zip(ALL_FEATURES, np.abs(_shap_arr).mean(axis=0).tolist()))
        _shap_baseline = {
            "mean_abs_shap": _mean_abs_shap,
            "n_samples":     _shap_sample_n,
            "features":      ALL_FEATURES,
        }
        shap_baseline_path = models_dir / "churn_shap_baseline.json"
        shap_baseline_path.write_text(json.dumps(_shap_baseline, indent=2))
        mlflow.log_artifact(str(shap_baseline_path))

        # 18c: OrdinalEncoder (joblib + SHA-256 checksum — detects tampering on load)
        encoder_path = models_dir / "churn_encoder.joblib"
        joblib.dump(encoder, encoder_path)
        checksum = hashlib.sha256(encoder_path.read_bytes()).hexdigest()
        (models_dir / "churn_encoder.sha256").write_text(checksum)
        mlflow.log_artifact(str(encoder_path))
        mlflow.log_artifact(str(models_dir / "churn_encoder.sha256"))

        # 18d: feature names JSON (dict — not list; Section 5 contract)
        feature_names_data = {
            "all_features":         ALL_FEATURES,
            "numeric_features":     NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "feature_count":        len(ALL_FEATURES),
        }
        fn_path = models_dir / "churn_feature_names.json"
        with open(fn_path, "w") as f:
            json.dump(feature_names_data, f, indent=2)
        mlflow.log_artifact(str(fn_path))

        # 18e-i: F1-optimised threshold (saved separately from baseline — load_churn_model reads this)
        threshold_path = models_dir / "churn_threshold.json"
        with open(threshold_path, "w") as f:
            json.dump({"threshold": float(optimal_threshold)}, f)
        if mlflow.active_run() is not None:
            mlflow.log_artifact(str(threshold_path))
        else:
            logger.debug("No active MLflow run — skipping artifact logging for threshold")

        # 18e-ii: baseline churn probabilities for Evidently drift (2000-row sample)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(
            len(df_test),
            size=min(BASELINE_SAMPLE_SIZE, len(df_test)),
            replace=False,
        )
        baseline_stats: dict[str, Any] = {
            "probabilities":         y_proba_test[sample_idx].tolist(),
            "labels":                y_test[sample_idx].tolist(),
            "feature_distributions": {
                col: X_test_df.iloc[sample_idx][col].tolist()
                for col in ALL_FEATURES
            },
            "threshold":      float(optimal_threshold),
            "churn_rate":     float(y_test.mean()),
            "n_samples":      int(len(sample_idx)),
            "n_test_total":   int(len(y_test)),
            "test_ref_date":  str(TEST_REF_DATE),
            "generated_at":   datetime.utcnow().isoformat(),
        }
        baseline_path = models_dir / "baseline_churn_proba.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline_stats, f, cls=_NumpyEncoder, indent=2)
        mlflow.log_artifact(str(baseline_path))

        mlflow.set_tag("mlflow.runName", "churn_xgboost")

        # Step 18f: versioned artifact directory + metadata + registry registration
        _version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _vdir = ARTIFACTS_DIR / "models" / "churn" / _version
        _vdir.mkdir(parents=True, exist_ok=True)
        _metadata = {
            "version":          _version,
            "model_name":       "churn",
            "trained_at":       datetime.utcnow().isoformat(),
            "roc_auc":          float(roc_auc),
            "f1_churn":         float(report["1"]["f1-score"]),
            "threshold":        float(optimal_threshold),
            "training_ref_date": str(TRAIN_REF_DATE),
            "feature_count":    len(ALL_FEATURES),
            "meets_threshold":  roc_auc >= config.CHURN_MIN_ROC_AUC,
        }
        (_vdir / "metadata.json").write_text(json.dumps(_metadata, indent=2))
        # Copy all artifacts into versioned directory
        for _fname in [
            "churn_model.json", "churn_explainer.json",
            "churn_encoder.joblib", "churn_encoder.sha256",
            "churn_threshold.json", "churn_quality.json",
            "churn_feature_names.json", "baseline_churn_proba.json",
        ]:
            _src = models_dir / _fname
            if _src.exists():
                import shutil as _shutil
                _shutil.copy2(_src, _vdir / _fname)

        # Register as "candidate" — DAG promotes after quality validation
        from src.model_registry.registry import ModelRecord, register
        register(ModelRecord(
            version=_version,
            model_name="churn",
            artifact_dir=f"models/churn/{_version}",
            status="candidate",
            trained_at=_metadata["trained_at"],
            roc_auc=float(roc_auc),
            threshold=float(optimal_threshold),
            feature_count=len(ALL_FEATURES),
        ))
        logger.info(
            "All churn artifacts saved to %s | versioned to %s | MLflow run_id=%s",
            models_dir, _vdir, run.info.run_id,
        )

    # Step 19: return bundle
    return ChurnModelBundle(
        model=model,
        explainer=explainer,
        encoder=encoder,
        feature_names=ALL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        threshold=optimal_threshold,
        churn_rate_train=float(y_train.mean()),
        churn_rate_test=float(y_test.mean()),
    )


def load_churn_model(version_dir: Path | None = None) -> ChurnModelBundle:
    """
    Loads churn model bundle from disk.
    Called once in FastAPI lifespan — never per-request.

    Resolution order for artifact directory:
        1. version_dir if explicitly supplied (e.g. by retraining DAG)
        2. Registry production entry  → artifacts/models/churn/<version>/
        3. Flat fallback              → artifacts/models/  (pre-registry layout)

    FastAPI inference contract:
        1. Feast online features → DataFrame (any column order)
        2. encoder.transform(df[categorical_features])
        3. X = df[feature_names]    # reorder AFTER encoding
        4. model.predict_proba(X)[:, 1]
    """
    if version_dir is None:
        from src.model_registry.registry import get_production_artifact_dir
        version_dir = get_production_artifact_dir("churn") or (ARTIFACTS_DIR / "models")

    models_dir = version_dir

    q_path = models_dir / "churn_quality.json"
    if q_path.exists():
        q = json.loads(q_path.read_text())
        if not q.get("meets_threshold", True):
            raise RuntimeError(
                f"Churn model ROC-AUC={q['roc_auc']:.3f} is below production threshold "
                f"{config.CHURN_MIN_ROC_AUC}. Retrain with more data."
            )

    model = xgb.XGBClassifier()
    model.load_model(str(models_dir / "churn_model.json"))

    explainer_path = models_dir / "churn_explainer.json"
    try:
        explainer: Any = shap.TreeExplainer.load(str(explainer_path))
    except (AttributeError, FileNotFoundError) as e:
        raise RuntimeError(
            "SHAP explainer load failed. Ensure shap>=0.44.0 and retrain the model."
        ) from e

    encoder_path  = models_dir / "churn_encoder.joblib"
    checksum_path = models_dir / "churn_encoder.sha256"
    if checksum_path.exists():
        expected = checksum_path.read_text().strip()
        actual   = hashlib.sha256(encoder_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Encoder checksum mismatch — possible file tampering detected")
    encoder: OrdinalEncoder = joblib.load(encoder_path)

    with open(models_dir / "churn_feature_names.json") as f:
        names: dict[str, Any] = json.load(f)

    # Load F1-optimised threshold; fall back to env-var config if artifact is missing
    threshold_path = models_dir / "churn_threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold = float(json.load(f)["threshold"])
        logger.info("Churn threshold loaded from artifact: %.4f", threshold)
    else:
        threshold = float(config.CHURN_THRESHOLD)
        logger.warning(
            "churn_threshold.json not found — using config default %.4f", threshold
        )

    logger.info("Churn model bundle loaded from %s", models_dir)
    return ChurnModelBundle(
        model=model,
        explainer=explainer,
        encoder=encoder,
        feature_names=names["all_features"],
        categorical_features=names["categorical_features"],
        threshold=threshold,
        churn_rate_train=0.0,   # not persisted — training-time only
        churn_rate_test=0.0,
    )
