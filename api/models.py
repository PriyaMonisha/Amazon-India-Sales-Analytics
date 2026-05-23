from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Churn
# ---------------------------------------------------------------------------

class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_predicted: bool
    threshold: float
    rfm_segment: str | None = None


class ShapContribution(BaseModel):
    feature: str
    shap_value: float


class ChurnExplainResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_predicted: bool
    threshold: float
    top_features: list[ShapContribution]   # top 10 sorted by |shap_value| descending


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

class ForecastPoint(BaseModel):
    month: str        # "YYYY-MM-DD"
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    subcategory: str
    periods: int
    forecast: list[ForecastPoint]


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

class PriceSuggestion(BaseModel):
    price_inr: float
    delta_pct: int       # -20, -10, 0, 10, 20
    demand_estimate: float
    revenue_estimate: float


class PricingResponse(BaseModel):
    subcategory: str
    current_price_inr: float
    current_demand_estimate: float
    month: int
    is_festival_month: bool
    price_suggestions: list[PriceSuggestion]


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class RecommendedItem(BaseModel):
    subcategory: str
    confidence: float | None
    lift: float | None
    source: str    # "association_rules" or "popularity_fallback"


class RecommendationResponse(BaseModel):
    input_subcategory: str
    top_n: int
    recommendations: list[RecommendedItem]


# ---------------------------------------------------------------------------
# Anomaly
# ---------------------------------------------------------------------------

class AnomalyRequest(BaseModel):
    final_amount_inr: float = Field(..., gt=0)
    mrp_inr: float = Field(..., gt=0)
    delivery_days: float = Field(..., ge=0)
    is_return: int = Field(default=0, ge=0, le=1)


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float                               # decision function; higher = more normal
    final_amount_inr: float
    mrp_inr: float
    discount_pct: float = Field(..., ge=0.0, le=1.0)  # clipped by detect_anomalies(); enforced here too


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
