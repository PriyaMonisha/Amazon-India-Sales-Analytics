from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from api.models import ForecastPoint, ForecastResponse
from src.models.forecasting import slug_from_subcategory

router = APIRouter(tags=["forecast"])
logger = logging.getLogger(__name__)


@router.get("/forecast/{subcategory}", response_model=ForecastResponse)
def predict_subcategory_forecast(subcategory: str, request: Request, periods: int = 3):
    if periods < 1 or periods > 12:
        raise HTTPException(422, detail="periods must be between 1 and 12")

    forecast_models: dict = request.app.state.models.get("forecast", {})
    slug = slug_from_subcategory(subcategory)

    if slug not in forecast_models:
        raise HTTPException(404, detail=f"No forecast model trained for '{subcategory}'")

    # Use pre-loaded Prophet object — do NOT call predict_forecast() which reloads from disk
    prophet_model = forecast_models[slug]
    future        = prophet_model.make_future_dataframe(periods=periods, freq="MS")
    full_forecast = prophet_model.predict(future)

    # Filter to future-only rows by date — .tail(periods) is unsafe on sparse training data
    last_train_date = prophet_model.history["ds"].max()
    forecast_df     = full_forecast[full_forecast["ds"] > last_train_date].head(periods)

    if len(forecast_df) < periods:
        raise HTTPException(
            500,
            detail=(
                f"Prophet returned {len(forecast_df)} future periods, expected {periods}. "
                "Training data may extend past the current date."
            ),
        )

    points = [
        ForecastPoint(
            month=str(row["ds"].date()),
            yhat=round(float(row["yhat"]), 2),
            yhat_lower=round(float(row["yhat_lower"]), 2),
            yhat_upper=round(float(row["yhat_upper"]), 2),
        )
        for _, row in forecast_df.iterrows()
    ]
    return ForecastResponse(subcategory=subcategory, periods=periods, forecast=points)
