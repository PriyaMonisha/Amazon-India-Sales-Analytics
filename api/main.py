from __future__ import annotations

import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

import config
from api.dependencies import limiter
from api.models import HealthResponse
from api.routers import anomaly, churn, forecast, pricing, recommendation
from src.models.anomaly import load_anomaly_model
from src.models.churn import load_churn_model
from src.models.forecasting import load_forecast_model, slug_from_subcategory
from src.models.pricing import load_pricing_model
from src.models.recommendation import load_recommendation_model
from src.monitoring.drift import flush_and_check
from src.monitoring.metrics import (
    MODEL_LOADED,
    PREDICTION_LATENCY_SECONDS,
    PREDICTION_REQUEST_COUNTER,
)

# Make src/ module logs visible alongside uvicorn logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus request tracking middleware
# Regex captures model name from URL: /predict/<model_name>/...
# MODEL_NAMES uses "recommend" (not "recommendation") — matches endpoint path and label values
# ---------------------------------------------------------------------------
_PREDICT_RE = re.compile(r"/predict/(churn|forecast|pricing|recommend|anomaly)")


class _PredictionMetricsMiddleware(BaseHTTPMiddleware):
    """Track latency and request count for every /predict/* endpoint."""

    async def dispatch(self, request: StarletteRequest, call_next):  # type: ignore[override]
        m = _PREDICT_RE.search(request.url.path)
        if not m:
            return await call_next(request)

        model_name = m.group(1)   # "recommend", not "recommendation"
        t0 = time.perf_counter()
        status = "success"
        try:
            response = await call_next(request)
            if response.status_code >= 400:
                status = "error"
        except Exception:
            status = "error"
            raise
        finally:
            elapsed = time.perf_counter() - t0
            PREDICTION_REQUEST_COUNTER.labels(model_name=model_name, status=status).inc()
            PREDICTION_LATENCY_SECONDS.labels(model_name=model_name).observe(elapsed)
        return response


# ---------------------------------------------------------------------------
# Lifespan: load all models, then set MODEL_LOADED Prometheus gauges
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    models: dict = {}

    # --- Churn ---
    try:
        models["churn"] = load_churn_model()
        logger.info("churn model loaded")
    except Exception as e:
        models["churn"] = None
        logger.warning("churn model not loaded: %s", e)

    # --- Forecast: load each Prophet model keyed by slug ---
    # config.ARTIFACTS_DIR is pathlib.WindowsPath — / operator works directly
    slugs_path = config.ARTIFACTS_DIR / "models" / "forecast_slugs.json"
    forecast_models: dict = {}
    try:
        if slugs_path.exists():
            subcats: list[str] = json.loads(slugs_path.read_text())["slugs"]
            for subcat in subcats:
                slug = slug_from_subcategory(subcat)
                # load_forecast_model takes the ORIGINAL subcategory name, returns Prophet object
                forecast_models[slug] = load_forecast_model(subcat)
            logger.info("forecast models loaded: %d subcategories", len(forecast_models))
        else:
            logger.warning("forecast_slugs.json not found — no Prophet models loaded")
    except Exception as e:
        logger.warning("forecast models not loaded: %s", e)
    models["forecast"] = forecast_models   # always a dict; may be empty {}

    # --- Pricing ---
    # Note: predict_optimal_price() also calls load_pricing_model() internally per request.
    # This preload serves as: (1) startup health signal, (2) fail-fast if file missing.
    try:
        models["pricing"] = load_pricing_model()
        logger.info("pricing model loaded")
    except Exception as e:
        models["pricing"] = None
        logger.warning("pricing model not loaded: %s", e)

    # --- Recommendation ---
    try:
        rules_df, fallback = load_recommendation_model()
        models["recommendation"] = {"rules": rules_df, "fallback": fallback}
        logger.info("recommendation model loaded")
    except Exception as e:
        models["recommendation"] = None
        logger.warning("recommendation model not loaded: %s", e)

    # --- Anomaly ---
    try:
        models["anomaly"] = load_anomaly_model()
        logger.info("anomaly model loaded")
    except Exception as e:
        models["anomaly"] = None
        logger.warning("anomaly model not loaded: %s", e)

    app.state.models = models

    # --- A/B config + optional challenger model ---
    ab_config_path = config.ARTIFACTS_DIR / ".." / "artifacts" / "ab_config.json"
    # Resolve relative path robustly: ab_config lives at project root / artifacts / ab_config.json
    ab_config_path = config.ARTIFACTS_DIR.parent / "artifacts" / "ab_config.json"
    # Simpler: ARTIFACTS_DIR is already the artifacts/ dir
    ab_config_path = config.ARTIFACTS_DIR / "ab_config.json"
    ab_cfg: dict | None = None
    if ab_config_path.exists():
        try:
            ab_cfg = json.loads(ab_config_path.read_text())
        except Exception as e:
            logger.warning("Could not load ab_config.json: %s", e)
    app.state.ab_config = ab_cfg

    # Load challenger churn model if A/B is enabled and a challenger version is set
    models["churn_challenger"] = None
    if ab_cfg and ab_cfg.get("enabled") and ab_cfg.get("challenger_version"):
        try:
            from config import ARTIFACTS_DIR
            challenger_dir = ARTIFACTS_DIR / "models" / "churn" / ab_cfg["challenger_version"]
            if challenger_dir.exists():
                models["churn_challenger"] = load_churn_model(version_dir=challenger_dir)
                logger.info(
                    "A/B challenger loaded: version=%s (%.0f%% traffic)",
                    ab_cfg["challenger_version"],
                    ab_cfg.get("challenger_traffic_pct", 0.10) * 100,
                )
            else:
                logger.warning("A/B challenger dir not found: %s", challenger_dir)
        except Exception as e:
            logger.warning("A/B challenger model not loaded: %s", e)

    # Set MODEL_LOADED gauges — label "recommend" matches regex capture and MODEL_NAMES constant
    MODEL_LOADED.labels(model_name="churn").set(1 if models.get("churn") is not None else 0)
    MODEL_LOADED.labels(model_name="forecast").set(1 if len(models.get("forecast", {})) > 0 else 0)
    MODEL_LOADED.labels(model_name="pricing").set(1 if models.get("pricing") is not None else 0)
    MODEL_LOADED.labels(model_name="recommend").set(1 if models.get("recommendation") is not None else 0)
    MODEL_LOADED.labels(model_name="anomaly").set(1 if models.get("anomaly") is not None else 0)

    yield
    # shutdown: in-memory objects — nothing to release


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Amazon India Sales Analytics API",
    description="ML inference: churn, forecast, pricing, recommendation, anomaly",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(_PredictionMetricsMiddleware)

app.include_router(churn.router,          prefix="/predict")
app.include_router(forecast.router,       prefix="/predict")
app.include_router(pricing.router,        prefix="/predict")
app.include_router(recommendation.router, prefix="/predict")
app.include_router(anomaly.router,        prefix="/predict")


# ---------------------------------------------------------------------------
# Ops endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request):
    m = request.app.state.models
    models_loaded = {
        "churn":          m.get("churn") is not None,
        "forecast":       len(m.get("forecast", {})) > 0,
        "pricing":        m.get("pricing") is not None,
        "recommendation": m.get("recommendation") is not None,
        "anomaly":        m.get("anomaly") is not None,
    }
    # Use "degraded" when no models are loaded so /health is self-explanatory
    # without needing to read the README (common recruiter experience)
    all_loaded = all(models_loaded.values())
    any_loaded = any(models_loaded.values())
    if all_loaded:
        status = "ok"
    elif any_loaded:
        status = "degraded — some models missing, run: make train"
    else:
        status = "degraded — no models loaded, run: make train first"
    return HealthResponse(status=status, models_loaded=models_loaded)


@app.get("/metrics", tags=["ops"])
def metrics():
    # MUST return text/plain — Prometheus scraper cannot parse JSON
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health/models", tags=["ops"])
def model_registry_health(request: Request) -> dict[str, Any]:
    """
    Returns production version metadata from the registry and A/B challenger status.
    Useful for ops dashboards and deployment verification.
    """
    from src.model_registry.registry import get_production
    return {
        "churn_production":       get_production("churn"),
        "pricing_production":     get_production("pricing"),
        "ab_enabled":             bool(
            getattr(request.app.state, "ab_config", None) and
            request.app.state.ab_config.get("enabled")
        ),
        "ab_challenger_loaded":   request.app.state.models.get("churn_challenger") is not None,
        "ab_challenger_version":  (
            request.app.state.ab_config.get("challenger_version")
            if getattr(request.app.state, "ab_config", None)
            else None
        ),
    }


@app.post("/monitor/drift/run", tags=["ops"])
@limiter.limit("10/minute")
def trigger_drift_check(request: Request) -> dict[str, Any]:
    """
    Drain the prediction feature buffer and run Evidently drift detection.
    Returns {} if buffer is empty or baseline_churn_proba.json is missing (pre-training).
    Prometheus gauges (DRIFT_KS_STATISTIC etc.) are updated as a side effect.
    """
    result = flush_and_check()
    if not result:
        return {"status": "skipped", "reason": "buffer_empty_or_no_baseline"}
    top: dict = result.get("metrics", [{}])[0].get("result", {})
    return {
        "status": "ok",
        "columns_checked": top.get("number_of_columns", 0),
        "columns_drifted": top.get("number_of_drifted_columns", 0),
        "dataset_drift":   top.get("dataset_drift", False),
    }
