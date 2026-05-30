import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# TestAppContract (2 tests)
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestAppContract:
    def test_models_schema(self, api_client_all_loaded):
        required = {"churn", "forecast", "anomaly", "pricing", "recommendation"}
        assert required.issubset(set(api_client_all_loaded.app.state.models.keys()))

    def test_forecast_keys_are_slugs(self, api_client_all_loaded):
        forecast = api_client_all_loaded.app.state.models["forecast"]
        assert all(k == k.lower() for k in forecast.keys()), \
            "Forecast model keys must be slugs (lowercase)"


# ---------------------------------------------------------------------------
# TestOpsEndpoints (5 tests)
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestOpsEndpoints:
    def test_health_returns_200_ok_when_all_models_loaded(self, api_client_all_loaded):
        r = api_client_all_loaded.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"   # "ok" only when ALL models loaded

    def test_health_models_loaded_are_bools(self, api_client_all_loaded):
        loaded = api_client_all_loaded.get("/health").json()["models_loaded"]
        assert all(isinstance(v, bool) for v in loaded.values()), \
            f"Non-bool values in models_loaded: {loaded}"

    def test_health_injected_models_show_all_true(self, api_client_all_loaded):
        loaded = api_client_all_loaded.get("/health").json()["models_loaded"]
        for model_name, is_loaded in loaded.items():
            assert is_loaded, f"Expected {model_name} to be loaded=True, got False"

    def test_health_no_models_shows_all_false(self, api_client_no_models):
        loaded = api_client_no_models.get("/health").json()["models_loaded"]
        true_models = [k for k, v in loaded.items() if v]
        assert not true_models, f"Un-injected models show True: {true_models}"

    def test_metrics_endpoint(self, api_client_all_loaded):
        r = api_client_all_loaded.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers.get("content-type", ""), \
            f"Expected text/plain, got: {r.headers.get('content-type')}"
        assert b"amazon_" in r.content, \
            "Custom amazon_ prefixed metrics not found in Prometheus output"


# ---------------------------------------------------------------------------
# TestAnomalyEndpoint (5 tests)
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestAnomalyEndpoint:
    def test_get_returns_405(self, api_client_all_loaded):
        r = api_client_all_loaded.get("/predict/anomaly")
        assert r.status_code == 405

    def test_valid_payload_200(self, api_client_all_loaded):
        r = api_client_all_loaded.post(
            "/predict/anomaly",
            json={"final_amount_inr": 500.0, "mrp_inr": 1000.0, "delivery_days": 3.0},
        )
        assert r.status_code == 200

    def test_discount_pct_correct(self, api_client_all_loaded):
        r = api_client_all_loaded.post(
            "/predict/anomaly",
            json={"final_amount_inr": 800.0, "mrp_inr": 1000.0, "delivery_days": 3.0},
        )
        assert r.status_code == 200
        assert r.json()["discount_pct"] == pytest.approx(0.2, abs=0.001)

    def test_zero_mrp_422(self, api_client_all_loaded):
        r = api_client_all_loaded.post(
            "/predict/anomaly",
            json={"final_amount_inr": 500.0, "mrp_inr": 0.0, "delivery_days": 3.0},
        )
        assert r.status_code == 422

    def test_503_no_bundle(self, api_client_no_models):
        r = api_client_no_models.post(
            "/predict/anomaly",
            json={"final_amount_inr": 500.0, "mrp_inr": 1000.0, "delivery_days": 3.0},
        )
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# TestChurnEndpoint (5 tests)
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestChurnEndpoint:
    def test_503_bundle_none(self, api_client_no_models):
        r = api_client_no_models.get("/predict/churn/CUST0001")
        assert r.status_code == 503

    def test_503_feast_empty(self, api_client_all_loaded):
        with patch("api.routers.churn.get_online_features", return_value=pd.DataFrame()):
            r = api_client_all_loaded.get("/predict/churn/CUST0001")
        assert r.status_code == 503

    def test_200_valid(self, api_client_all_loaded, mock_feast_features):
        with patch("api.routers.churn.get_online_features", return_value=mock_feast_features):
            r = api_client_all_loaded.get("/predict/churn/CUST0001")
        assert r.status_code == 200
        data = r.json()
        assert data["customer_id"] == "CUST0001"
        assert isinstance(data["churn_probability"], float)
        assert 0.0 <= data["churn_probability"] <= 1.0, \
            f"churn_probability={data['churn_probability']} outside [0,1]"
        assert isinstance(data["churn_predicted"], bool)

    def test_explain_200_list_shap(self, api_client_all_loaded, mock_feast_features):
        # SHAP mock returns [neg_class, pos_class] list — exercises Rule 45 sv[1][0] guard
        with patch("api.routers.churn.get_online_features", return_value=mock_feast_features):
            r = api_client_all_loaded.get("/predict/churn/CUST0001/explain")
        assert r.status_code == 200

    def test_explain_200_ndarray_shap(self, api_client_all_loaded, mock_feast_features):
        # Tests the non-list (ndarray) SHAP branch — Rule 45.
        # raise_server_exceptions=True is intentional: if the ndarray guard is missing,
        # pytest shows ERROR with the router traceback (not FAILED with assertion mismatch).
        # That's the most informative failure mode — traceback points directly to the bug.
        from src.models.churn import ALL_FEATURES
        bundle = api_client_all_loaded.app.state.models["churn"]
        bundle["explainer"].shap_values.return_value = np.zeros((1, len(ALL_FEATURES)))
        with patch("api.routers.churn.get_online_features", return_value=mock_feast_features):
            r = api_client_all_loaded.get("/predict/churn/CUST0001/explain")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# TestForecastEndpoint (5 tests)
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestForecastEndpoint:
    def test_404_unknown_subcategory(self, api_client_all_loaded):
        r = api_client_all_loaded.get("/predict/forecast/nonexistent_category?periods=3")
        assert r.status_code == 404

    def test_422_periods_too_high(self, api_client_all_loaded):
        r = api_client_all_loaded.get("/predict/forecast/smartphones?periods=13")
        assert r.status_code == 422

    def test_422_periods_zero(self, api_client_all_loaded):
        r = api_client_all_loaded.get("/predict/forecast/smartphones?periods=0")
        assert r.status_code == 422

    def test_200_verifies_response_body(self, api_client_all_loaded):
        r = api_client_all_loaded.get("/predict/forecast/smartphones?periods=3")
        assert r.status_code == 200
        data = r.json()
        assert len(data["forecast"]) == 3
        for pt in data["forecast"]:
            assert {"month", "yhat", "yhat_lower", "yhat_upper"}.issubset(pt)

    def test_slug_conversion_mixed_case_url(self, api_client_all_loaded):
        # "Smartphones" → slug_from_subcategory → "smartphones" → finds fixture key
        r = api_client_all_loaded.get("/predict/forecast/Smartphones?periods=3")
        assert r.status_code == 200  # 404 if router doesn't slugify URL param


# ---------------------------------------------------------------------------
# TestPricingEndpoint (4 tests)
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestPricingEndpoint:
    def test_503_when_none(self, api_client_no_models):
        r = api_client_no_models.get("/predict/pricing/Smartphones?current_price=20000")
        assert r.status_code == 503

    def test_200_valid(self, api_client_all_loaded, mock_pricing_result):
        with patch("api.routers.pricing.predict_optimal_price", return_value=mock_pricing_result):
            r = api_client_all_loaded.get("/predict/pricing/Smartphones?current_price=20000")
        assert r.status_code == 200
        assert len(r.json()["price_suggestions"]) == 5

    def test_422_zero_price(self, api_client_all_loaded):
        r = api_client_all_loaded.get("/predict/pricing/Smartphones?current_price=0")
        assert r.status_code == 422

    def test_five_delta_values(self, api_client_all_loaded, mock_pricing_result):
        with patch("api.routers.pricing.predict_optimal_price", return_value=mock_pricing_result):
            r = api_client_all_loaded.get("/predict/pricing/Smartphones?current_price=20000")
        suggestions = r.json()["price_suggestions"]
        assert "delta_pct" in suggestions[0], \
            f"Expected 'delta_pct' field. Got: {list(suggestions[0].keys())}"
        deltas = sorted(s["delta_pct"] for s in suggestions)
        assert deltas == [-20, -10, 0, 10, 20]


# ---------------------------------------------------------------------------
# TestRecommendEndpoint (4 tests)
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestRecommendEndpoint:
    def test_503_when_none(self, api_client_no_models):
        r = api_client_no_models.get("/predict/recommend/Laptops?top_n=3")
        assert r.status_code == 503

    def test_200_valid(self, api_client_all_loaded):
        r = api_client_all_loaded.get("/predict/recommend/Laptops?top_n=3")
        assert r.status_code == 200
        data = r.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) <= 3, \
            f"top_n=3 but got {len(data['recommendations'])} recommendations"

    def test_422_top_n_too_high(self, api_client_all_loaded):
        r = api_client_all_loaded.get("/predict/recommend/Laptops?top_n=51")
        assert r.status_code == 422

    def test_422_top_n_zero(self, api_client_all_loaded):
        r = api_client_all_loaded.get("/predict/recommend/Laptops?top_n=0")
        assert r.status_code == 422
