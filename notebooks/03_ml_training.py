# %% [markdown]
# # Section 4: ML Model Training
#
# ⚠️  **START Docker Desktop before running this notebook.**
# PostgreSQL must be accessible on localhost:5433.
#
# After training, **STOP Docker Desktop**: `docker compose down`, then close the app.
#
# Run order:
# 1. Start Docker Desktop
# 2. `docker compose up -d postgres` (PostgreSQL only — Redis/MLflow optional for offline training)
# 3. Run this notebook cell by cell
# 4. Verify artifacts in `artifacts/models/`

# %% [markdown]
# ## 0 — Setup

# %%
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import DB_URL, ARTIFACTS_DIR, MLFLOW_TRACKING_URI, FAST_MODE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("03_ml_training")

from sqlalchemy import create_engine

engine = create_engine(DB_URL)

# Quick connectivity check
with engine.connect() as conn:
    result = conn.execute(__import__("sqlalchemy").text("SELECT COUNT(*) FROM fact_transactions")).scalar()
    logger.info("Connected to PostgreSQL | fact_transactions rows: %d", result)

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
logger.info("FAST_MODE=%s | MLFLOW_TRACKING_URI=%s", FAST_MODE, MLFLOW_TRACKING_URI)

# %% [markdown]
# ## 1 — Churn Model (XGBoost + Optuna + SHAP)

# %%
from src.models.churn import train_churn_model, load_churn_model

logger.info("=== Training Churn Model ===")
churn_bundle = train_churn_model(engine)

print("\n--- Churn Model Results ---")
print(f"  ROC-AUC:          {churn_bundle['churn_rate_train']:.4f}")
print(f"  Threshold:        {churn_bundle['threshold']:.4f}")
print(f"  Churn rate train: {churn_bundle['churn_rate_train']:.1%}")
print(f"  Churn rate test:  {churn_bundle['churn_rate_test']:.1%}")
print(f"  Features:         {len(churn_bundle['feature_names'])}")

# Verify load
logger.info("Verifying churn model load...")
loaded_churn = load_churn_model()
assert loaded_churn["model"] is not None
assert loaded_churn["explainer"] is not None
logger.info("Churn model load OK")

# %% [markdown]
# ## 2 — Forecasting Models (Prophet per subcategory)

# %%
from src.models.forecasting import train_forecast_models, load_forecast_model, predict_forecast, list_trained_subcategories

import numpy as np

logger.info("=== Training Forecast Models ===")
wmape_by_slug = train_forecast_models(engine)

print("\n--- Forecast Model Results ---")
print(f"  Subcategories trained: {len(wmape_by_slug)}")
if wmape_by_slug:
    wmapes = list(wmape_by_slug.values())
    print(f"  Mean WMAPE:  {np.mean(wmapes):.3f}")
    print(f"  Median WMAPE: {np.median(wmapes):.3f}")
    print(f"  Best:  {min(wmape_by_slug, key=wmape_by_slug.get)} ({min(wmapes):.3f})")
    print(f"  Worst: {max(wmape_by_slug, key=wmape_by_slug.get)} ({max(wmapes):.3f})")

# Verify: load and predict for first trained subcategory
trained_slugs = list_trained_subcategories()
if trained_slugs:
    sample_slug = trained_slugs[0]
    forecast_df = predict_forecast(sample_slug, periods=3)
    logger.info("Sample forecast for '%s': %d rows", sample_slug, len(forecast_df))
    print(f"\n  Sample forecast ({sample_slug}):")
    print(forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_string(index=False))

# %% [markdown]
# ## 3 — Pricing Model (XGBoost Elasticity)

# %%
from src.models.pricing import train_pricing_model, load_pricing_model, predict_optimal_price

logger.info("=== Training Pricing Model ===")
pricing_result = train_pricing_model(engine)

print("\n--- Pricing Model Results ---")
print(f"  R²:   {pricing_result['r2']:.4f}")
print(f"  RMSE: {pricing_result['rmse']:.4f} (log-scale)")

# Verify load + sample prediction
loaded_pricing = load_pricing_model()
assert loaded_pricing["model"] is not None

# Sample pricing prediction (use a generic subcategory name — actual subcategory TBD at runtime)
logger.info("Pricing model load OK")

# %% [markdown]
# ## 4 — Recommendation Model (FP-Growth)

# %%
from src.models.recommendation import train_recommendation_model, load_recommendation_model, get_recommendations

logger.info("=== Training Recommendation Model ===")
rules_df = train_recommendation_model(engine)

print("\n--- Recommendation Model Results ---")
print(f"  Association rules: {len(rules_df)}")
if not rules_df.empty:
    print(f"  Max lift: {rules_df['lift'].max():.3f}")
    print(f"  Mean lift: {rules_df['lift'].mean():.3f}")

# Verify load
loaded_rules, loaded_fallback = load_recommendation_model()
global_top = loaded_fallback.get("global_top", [])
print(f"  Popularity fallback: {len(global_top)} subcategories")
if global_top:
    print(f"  Top 3 global: {global_top[:3]}")

# Sample recommendations
if global_top:
    sample_subcat = global_top[0]
    recs = get_recommendations(sample_subcat, top_n=5, rules=loaded_rules, fallback=loaded_fallback)
    print(f"\n  Sample recs for '{sample_subcat}':")
    for r in recs:
        src = r["source"]
        lift = f"lift={r['lift']:.2f}" if r.get("lift") else "popularity"
        print(f"    → {r['subcategory']} ({lift}) [{src}]")

# %% [markdown]
# ## 5 — Anomaly Detection (IsolationForest)

# %%
from src.models.anomaly import train_anomaly_model, load_anomaly_model, detect_anomalies

logger.info("=== Training Anomaly Model ===")
anomaly_result = train_anomaly_model(engine)

print("\n--- Anomaly Model Results ---")
print(f"  Anomalies detected: {anomaly_result['n_anomalies']}")
print(f"  Anomaly rate:       {anomaly_result['anomaly_rate']:.2%}")

# Verify load
loaded_anomaly = load_anomaly_model()
assert loaded_anomaly["model"] is not None
logger.info("Anomaly model load OK")

# %% [markdown]
# ## 6 — Verification Summary

# %%
from pathlib import Path

artifacts_dir = ARTIFACTS_DIR / "models"
expected_files = [
    "churn_model.json",
    "churn_explainer.pkl",
    "churn_encoder.pkl",
    "churn_feature_names.json",
    "baseline_churn_proba.json",
    "forecast_slugs.json",
    "pricing_model.json",
    "pricing_encoder.pkl",
    "pricing_meta.json",
    "association_rules.parquet",
    "popularity_fallback.json",
    "anomaly_model.pkl",
    "anomaly_scaler.pkl",
]

print("\n=== Artifact Verification ===")
all_ok = True
for fname in expected_files:
    fpath = artifacts_dir / fname
    status = "✓" if fpath.exists() else "✗ MISSING"
    size   = f"({fpath.stat().st_size / 1024:.1f} KB)" if fpath.exists() else ""
    print(f"  {status:10s} {fname} {size}")
    if not fpath.exists():
        all_ok = False

# Forecast models
slugs = list_trained_subcategories()
for slug in slugs[:3]:
    fpath = artifacts_dir / f"forecast_{slug}.json"
    status = "✓" if fpath.exists() else "✗ MISSING"
    print(f"  {status:10s} forecast_{slug}.json")

if all_ok:
    print("\n✓ All required artifacts present. Section 4 complete.")
else:
    print("\n✗ Some artifacts missing — check logs above.")

print(f"\n  MLflow UI: {MLFLOW_TRACKING_URI}")
print("  Run: `mlflow ui --backend-store-uri sqlite:///mlflow.db` to view experiments")
