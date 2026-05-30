from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.dependencies import verify_api_key
from api.models import AnomalyRequest, AnomalyResponse
from src.models.anomaly import detect_anomalies

router = APIRouter(tags=["anomaly"], dependencies=[Depends(verify_api_key)])
logger = logging.getLogger(__name__)


@router.post("/anomaly", response_model=AnomalyResponse)
def predict_anomaly(body: AnomalyRequest, request: Request):
    bundle = request.app.state.models.get("anomaly")
    if bundle is None:
        raise HTTPException(503, detail="Anomaly model not loaded")

    # detect_anomalies() calls _build_features() internally:
    #   discount_pct = clip(1 - final_amount_inr / mrp_inr, 0, 1)
    # Both columns must be present so the internal computation works.
    input_df = pd.DataFrame([{
        "final_amount_inr": body.final_amount_inr,
        "mrp_inr":          body.mrp_inr,
        "delivery_days":    body.delivery_days,
        "is_return":        body.is_return,
    }])

    result_df = detect_anomalies(
        input_df,
        model=bundle["model"],
        scaler=bundle["scaler"],
    )
    row = result_df.iloc[0]

    # Read discount_pct from detect_anomalies() output — computed with same clip formula
    # as training. Do NOT recompute independently to avoid formula drift.
    return AnomalyResponse(
        is_anomaly=bool(row["is_anomaly"]),
        anomaly_score=round(float(row["anomaly_score"]), 6),
        final_amount_inr=body.final_amount_inr,
        mrp_inr=body.mrp_inr,
        discount_pct=round(float(row["discount_pct"]), 4),
    )
