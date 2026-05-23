from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from api.models import RecommendedItem, RecommendationResponse
from src.models.recommendation import get_recommendations

router = APIRouter(tags=["recommendation"])
logger = logging.getLogger(__name__)


@router.get("/recommend/{subcategory}", response_model=RecommendationResponse)
def recommend(subcategory: str, request: Request, top_n: int = 5):
    if top_n < 1 or top_n > 50:
        raise HTTPException(422, detail="top_n must be between 1 and 50")

    rec_bundle = request.app.state.models.get("recommendation")
    if rec_bundle is None:
        raise HTTPException(503, detail="Recommendation model not loaded")

    # Verified signature: get_recommendations(subcategory, top_n=5, rules=None, fallback=None)
    recs = get_recommendations(
        subcategory=subcategory,
        top_n=top_n,
        rules=rec_bundle["rules"],
        fallback=rec_bundle["fallback"],
    )
    return RecommendationResponse(
        input_subcategory=subcategory,
        top_n=top_n,
        recommendations=[RecommendedItem(**r) for r in recs],
    )
