from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.models import ChurnExplainResponse, ChurnResponse, ShapContribution
from src.features.feature_store import get_online_features
from src.models.churn import NUMERIC_FEATURES          # confirmed at churn.py:46
from src.monitoring.drift import maybe_run_drift_check, record_churn_features
from src.monitoring.metrics import CHURN_PROBABILITY_HISTOGRAM

router = APIRouter(tags=["churn"])
logger = logging.getLogger(__name__)


def _get_churn_features(customer_id: str) -> pd.DataFrame:
    df = get_online_features([customer_id])
    if df.empty:
        raise HTTPException(
            503,
            detail="Feature store unavailable or customer not materialized in Redis",
        )
    return df


def _encode_and_predict(df: pd.DataFrame, bundle: dict) -> tuple[float, np.ndarray]:
    """
    Encode categorical features IN-PLACE, then reorder to training column order, then predict.

    Returns (churn_probability, X) where X has shape (1, n_features).

    customer_id from Feast is excluded implicitly — it is not in feat_names (Rule 41).
    """
    cat_feats  = bundle["categorical_features"]
    feat_names = bundle["feature_names"]
    encoder    = bundle["encoder"]
    model      = bundle["model"]

    df = df.copy()

    # Rule 42: is_prime_member from Feast may be bool or NaN
    if "is_prime_member" in df.columns:
        df["is_prime_member"] = df["is_prime_member"].fillna(0).astype(int)

    # Step 1: encode categorical columns IN-PLACE
    df[cat_feats] = encoder.transform(df[cat_feats])

    # Step 2: reorder to match training contract AFTER encoding
    X    = df[feat_names].values           # shape (1, 14)
    prob = float(model.predict_proba(X)[0, 1])
    return prob, X


@router.get("/churn/{customer_id}", response_model=ChurnResponse)
def predict_churn(customer_id: str, request: Request):
    bundle = request.app.state.models.get("churn")
    if bundle is None:
        raise HTTPException(503, detail="Churn model not loaded")

    df = _get_churn_features(customer_id)
    prob, _ = _encode_and_predict(df, bundle)
    threshold = bundle["threshold"]

    # Monitoring — non-fatal: must never break inference
    try:
        CHURN_PROBABILITY_HISTOGRAM.observe(prob)
        numeric_row = {col: float(df[col].iloc[0]) for col in NUMERIC_FEATURES if col in df.columns}
        record_churn_features(numeric_row)
        maybe_run_drift_check()   # thread-safe; auto-fires when buffer >= 100
    except Exception as e:
        logger.warning("monitoring error (non-fatal): %s", e)

    return ChurnResponse(
        customer_id=customer_id,
        churn_probability=round(prob, 6),
        churn_predicted=prob >= threshold,
        threshold=threshold,
        rfm_segment=str(df["rfm_segment"].iloc[0]) if "rfm_segment" in df.columns else None,
    )


@router.get("/churn/{customer_id}/explain", response_model=ChurnExplainResponse)
def explain_churn(customer_id: str, request: Request):
    bundle = request.app.state.models.get("churn")
    if bundle is None:
        raise HTTPException(503, detail="Churn model not loaded")

    df = _get_churn_features(customer_id)
    prob, X = _encode_and_predict(df, bundle)

    explainer  = bundle["explainer"]
    feat_names = bundle["feature_names"]
    threshold  = bundle["threshold"]

    # Rule 44 — MANDATORY SHAP guard:
    # shap==0.44.0 + XGBClassifier may return list [neg_class, pos_class] OR single 2D array.
    # sv[0] on a list = negative class SHAP = wrong sign = backwards explanations.
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        row_shap = sv[1][0]    # positive class, first row → shape (14,)
    else:
        row_shap = sv[0]       # single 2D array, first row → shape (14,)

    contributions = [
        ShapContribution(feature=feat_names[i], shap_value=float(row_shap[i]))
        for i in range(len(feat_names))
    ]
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

    return ChurnExplainResponse(
        customer_id=customer_id,
        churn_probability=round(prob, 6),
        churn_predicted=prob >= threshold,
        threshold=threshold,
        top_features=contributions[:10],
    )
