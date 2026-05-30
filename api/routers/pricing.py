from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from api.dependencies import verify_api_key
from api.models import PriceSuggestion, PricingResponse
from src.models.pricing import predict_optimal_price

router = APIRouter(tags=["pricing"], dependencies=[Depends(verify_api_key)])
logger = logging.getLogger(__name__)

@router.get("/pricing/{subcategory}", response_model=PricingResponse)
def pricing_prediction(subcategory: str, current_price: float, request: Request):
    bundle = request.app.state.models.get("pricing")
    if bundle is None:
        raise HTTPException(503, detail="Pricing model not loaded")
    if current_price <= 0:
        raise HTTPException(422, detail="current_price must be positive")

    result = predict_optimal_price(
        subcategory=subcategory,
        current_price=current_price,
        bundle=bundle,     # pass preloaded bundle — zero disk I/O per request
    )

    # Explicitly construct nested objects — Pydantic v2 coerces dicts automatically,
    # but explicit construction catches key-name mismatches at call time not serialize time.
    result["price_suggestions"] = [
        PriceSuggestion(**ps) for ps in result["price_suggestions"]
    ]
    return PricingResponse(**result)
