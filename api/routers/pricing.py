from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from api.models import PriceSuggestion, PricingResponse
from src.models.pricing import predict_optimal_price

router = APIRouter(tags=["pricing"])
logger = logging.getLogger(__name__)

# NOTE on pricing preload vs per-request reload:
# lifespan preloads the pricing bundle into app.state.models["pricing"] for two reasons:
#   1. Startup health signal: models["pricing"] is not None = file loadable at boot
#   2. Fail-fast: missing artifact shows False in /health before first request
# predict_optimal_price() calls load_pricing_model() internally on every call (~5ms).
# This is accepted for portfolio. Refactor for prod: extract logic to accept a bundle arg.


@router.get("/pricing/{subcategory}", response_model=PricingResponse)
def pricing_prediction(subcategory: str, current_price: float, request: Request):
    # mrp is NOT a parameter — model works with relative price deltas only
    # 503 guard checks model was loadable at startup (not that this call uses the bundle)
    if request.app.state.models.get("pricing") is None:
        raise HTTPException(503, detail="Pricing model not loaded")
    if current_price <= 0:
        raise HTTPException(422, detail="current_price must be positive")

    result = predict_optimal_price(subcategory=subcategory, current_price=current_price)

    # Explicitly construct nested objects — Pydantic v2 coerces dicts automatically,
    # but explicit construction catches key-name mismatches at call time not serialize time.
    result["price_suggestions"] = [
        PriceSuggestion(**ps) for ps in result["price_suggestions"]
    ]
    return PricingResponse(**result)
