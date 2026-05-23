import json
import logging
import pickle
from datetime import date, datetime, timedelta
from typing import Any

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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing_extensions import TypedDict

import config
from config import (
    ARTIFACTS_DIR,
    CV_FOLDS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    N_OPTUNA_TRIALS,
    RANDOM_STATE,
    SAMPLE_ROWS,
)

logger = logging.getLogger(__name__)

# --- Constants ---
TRAIN_REF_DATE    = date(2023, 1, 1)   # features < 2023-01-01; label Jan–Mar 2023
TEST_REF_DATE     = date(2024, 1, 1)   # features < 2024-01-01; label Jan–Mar 2024
CHURN_WINDOW_DAYS = 90
DATASET_END_DATE  = date(2025, 12, 31)
BASELINE_SAMPLE_N = 2000  # rows sampled from test set for Evidently reference JSON

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
        (:ref_date::date - MAX(ft.order_date)::date)      AS days_since_last_purchase,
        COUNT(*) FILTER (
            WHERE ft.order_date >= :ref_date::date - INTERVAL '90 days'
        )                                                  AS total_orders_90d,
        COUNT(*)                                           AS total_orders_all_time,
        AVG(ft.final_amount_inr) FILTER (
            WHERE ft.order_date >= :ref_date::date - INTERVAL '180 days'
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
    WHERE ft.order_date < :ref_date::date
    GROUP BY ft.customer_id
),
label_window AS (
    SELECT customer_id
    FROM fact_transactions
    WHERE order_date >= :ref_date::date
      AND order_date < :obs_end::date
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


# --- MLflow setup ---

def _setup_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)


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

    # Step 4: churn rate sanity check
    train_churn_rate = float(df_train["churned"].mean())
    assert 0.05 < train_churn_rate < 0.70, (
        f"Unexpected churn rate {train_churn_rate:.1%} for TRAIN set. "
        "Check label_window SQL — likely INNER JOIN dropped churned customers, "
        "wrong obs_end date, or ref_date outside dataset range."
    )

    # Step 5: sort for reproducible StratifiedKFold folds
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
    _setup_mlflow("amazon_churn")
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
                "eval_metric":       "auc",
                "tree_method":       "hist",
            }
            clf = xgb.XGBClassifier(**params)
            cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=False)
            scores = cross_val_score(
                clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
            )
            return float(scores.mean())

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

        best_params: dict[str, Any] = dict(study.best_params)
        best_params.update({
            "scale_pos_weight": scale_pos_weight,
            "random_state":     RANDOM_STATE,
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

        # 18a: XGBoost model (native JSON — not pickle)
        model_path = models_dir / "churn_model.json"
        model.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))

        # 18b: SHAP TreeExplainer (pickled — pre-loaded once in FastAPI lifespan)
        explainer = shap.TreeExplainer(model)
        explainer_path = models_dir / "churn_explainer.pkl"
        with open(explainer_path, "wb") as f:
            pickle.dump(explainer, f)
        mlflow.log_artifact(str(explainer_path))

        # 18c: OrdinalEncoder (pickled)
        encoder_path = models_dir / "churn_encoder.pkl"
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
        mlflow.log_artifact(str(encoder_path))

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

        # 18e: baseline churn probabilities for Evidently drift (2000-row sample)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(
            len(df_test),
            size=min(BASELINE_SAMPLE_N, len(df_test)),
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
            "generated_at":   datetime.now().isoformat(),
        }
        baseline_path = models_dir / "baseline_churn_proba.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline_stats, f, cls=_NumpyEncoder, indent=2)
        mlflow.log_artifact(str(baseline_path))

        mlflow.set_tag("mlflow.runName", "churn_xgboost")
        logger.info(
            "All churn artifacts saved to %s | MLflow run_id=%s",
            models_dir, run.info.run_id,
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


def load_churn_model() -> ChurnModelBundle:
    """
    Loads churn model bundle from disk.
    Called once in FastAPI lifespan — never per-request.

    FastAPI inference contract:
        1. Feast online features → DataFrame (any column order)
        2. encoder.transform(df[categorical_features])
        3. X = df[feature_names]    # reorder AFTER encoding
        4. model.predict_proba(X)[:, 1]
    """
    models_dir = ARTIFACTS_DIR / "models"

    model = xgb.XGBClassifier()
    model.load_model(str(models_dir / "churn_model.json"))

    with open(models_dir / "churn_explainer.pkl", "rb") as f:
        explainer: Any = pickle.load(f)

    with open(models_dir / "churn_encoder.pkl", "rb") as f:
        encoder: OrdinalEncoder = pickle.load(f)

    with open(models_dir / "churn_feature_names.json") as f:
        names: dict[str, Any] = json.load(f)

    logger.info("Churn model bundle loaded from %s", models_dir)
    return ChurnModelBundle(
        model=model,
        explainer=explainer,
        encoder=encoder,
        feature_names=names["all_features"],
        categorical_features=names["categorical_features"],
        threshold=float(config.CHURN_THRESHOLD),  # can be overridden via env var
        churn_rate_train=0.0,   # not persisted — training-time only
        churn_rate_test=0.0,
    )
