import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixture Lifecycle:
#
# SESSION scope (shared across entire pytest run):
#   _isolate_prometheus      — autouse: snapshots/restores Prometheus registry
#   trained_anomaly_bundle   — IsolationForest + StandardScaler, trained once
#
# FUNCTION scope (fresh per test — prevents state leaks):
#   sample_sales_df          — 100-row sales DataFrame with valid values
#   sample_products_df       — 50-row product catalog
#   mock_engine              — MagicMock SQLAlchemy engine (begin + connect)
#   mock_anomaly_bundle      — wraps trained_anomaly_bundle into API-expected dict
#   mock_feast_features      — 1-row DataFrame with all 14 churn features
#   mock_pricing_result      — fresh dict copy per test (prevents in-place mutation)
#   api_client_all_loaded    — TestClient with all 5 models in app.state
#   api_client_no_models     — TestClient with all models set to None/{}
#
# Module-level helpers (not fixtures — called directly):
#   _make_minimal_valid_sales_df(n)  — for validation tests
#   _make_churn_bundle()             — called by api_client_all_loaded
#   _make_prophet_mock()             — called by api_client_all_loaded
#   _make_pricing_bundle()           — called by api_client_all_loaded
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_minimal_valid_sales_df(n: int = 5) -> pd.DataFrame:
    """All values in valid ranges. Mutate one column per test to inject invalid values."""
    return pd.DataFrame({
        "transaction_id":   [f"TXN{i:06d}" for i in range(n)],
        "customer_id":      [f"CUST{i:04d}" for i in range(n)],
        "product_id":       [f"PROD{i:04d}" for i in range(n)],
        "order_date":       pd.date_range("2022-01-01", periods=n, freq="D"),
        "final_amount_inr": [1000.0] * n,
        "mrp_inr":          [1500.0] * n,
        "delivery_days":    [3.0] * n,
        "customer_rating":  [4.0] * n,
        "customer_city":    ["Mumbai"] * n,
        "payment_method":   ["UPI"] * n,
        "is_prime_member":  [True] * n,
        "subcategory":      ["Smartphones"] * n,
        "category":         ["Electronics"] * n,
        "return_status":    ["Not Returned"] * n,
    })


def _make_churn_bundle():
    from src.models.churn import ALL_FEATURES, CATEGORICAL_FEATURES
    import xgboost as xgb
    from sklearn.preprocessing import OrdinalEncoder
    rng = np.random.default_rng(42)
    n = 20
    encoder = OrdinalEncoder(
        categories=[
            ["At Risk", "Champion", "Loyal", "Lost", "New"],
            ["Books", "Electronics", "Fashion", "Home"],
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(pd.DataFrame({
        "rfm_segment":        ["Champion", "Loyal", "At Risk", "Lost", "New"],
        "preferred_category": ["Electronics", "Home", "Fashion", "Books", "Electronics"],
    }))
    X_df = pd.DataFrame(rng.random((n, len(ALL_FEATURES))), columns=ALL_FEATURES)
    for col in CATEGORICAL_FEATURES:
        X_df[col] = rng.choice(["Champion", "Loyal", "At Risk"], n)
    y = np.array([0, 1] * 10)
    X_enc = X_df.copy()
    X_enc[CATEGORICAL_FEATURES] = encoder.transform(X_df[CATEGORICAL_FEATURES])
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X_enc[ALL_FEATURES].values, y)
    explainer = MagicMock()
    # List form [neg_class, pos_class] — exercises Rule 45 sv[1][0] guard
    explainer.shap_values.return_value = [
        np.zeros((1, len(ALL_FEATURES))),
        np.zeros((1, len(ALL_FEATURES))),
    ]
    return {
        "model":                model,
        "explainer":            explainer,
        "encoder":              encoder,
        "feature_names":        ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "threshold":            0.5,
        "churn_rate_train":     0.25,
        "churn_rate_test":      0.27,
    }


def _make_prophet_mock():
    history_dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    last_hist = history_dates[-1]
    mock = MagicMock()
    mock.history = pd.DataFrame({"ds": history_dates})  # real DataFrame for .max()

    def make_future(periods, freq="MS"):
        # [1:] removes last_hist → exactly `periods` future dates
        future = pd.date_range(last_hist, periods=periods + 1, freq=freq)[1:]
        return pd.DataFrame({"ds": list(history_dates) + list(future)})

    def predict(future_df):
        n = len(future_df)  # real int — router's len() check works
        return pd.DataFrame({
            "ds":         future_df["ds"].values,
            "yhat":       [100000.0] * n,
            "yhat_lower": [80000.0]  * n,
            "yhat_upper": [120000.0] * n,
        })

    mock.make_future_dataframe.side_effect = make_future
    mock.predict.side_effect = predict
    return mock


def _make_pricing_bundle():
    return {"loaded": True}  # non-None: passes 503 guard in pricing router


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _isolate_prometheus():
    """Unregister metrics added during session.
    No-op in standard single-run pytest. Guards IDE reruns / pytest-watch."""
    from prometheus_client import REGISTRY

    def _snapshot():
        named = set(REGISTRY._names_to_collectors.values())
        unnamed = set(REGISTRY._collectors_without_names)
        return named | unnamed

    collectors_before = _snapshot()
    yield
    for collector in _snapshot() - collectors_before:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture(scope="session")
def trained_anomaly_bundle():
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(42)
    n = 100
    X = np.column_stack([
        rng.uniform(100, 50000, n),
        rng.uniform(0, 0.8, n),
        rng.uniform(0, 14, n),
        rng.choice([0, 1], n).astype(float),
    ])
    scaler = StandardScaler().fit(X)
    model  = IsolationForest(contamination=0.02, random_state=42).fit(scaler.transform(X))
    return model, scaler


# ---------------------------------------------------------------------------
# Function-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_sales_df():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "transaction_id":   [f"TXN{i:05d}" for i in range(n)],     # all unique
        "customer_id":      [f"CUST{i:04d}" for i in range(n)],
        "product_id":       [f"PROD{i % 20:04d}" for i in range(n)],
        "order_date":       pd.date_range("2022-01-01", periods=n, freq="D"),
        "final_amount_inr": rng.uniform(100, 50000, n),
        "mrp_inr":          rng.uniform(200, 60000, n),
        "delivery_days":    rng.integers(0, 15, n).astype(float),
        "customer_rating":  rng.integers(1, 6, n).astype(float),  # 1–5, NOT 0–1
        "customer_city":    rng.choice(["Mumbai", "Delhi", "Bangalore"], n),
        "payment_method":   rng.choice(["UPI", "COD", "Credit Card", "Debit Card"], n),
        "is_prime_member":  rng.choice([True, False], n),
        "subcategory":      rng.choice(["Smartphones", "Laptops", "Smart TVs"], n),
        "category":         rng.choice(["Electronics", "Home"], n),
        "return_status":    rng.choice(["Returned", "Not Returned"], n),
    })


@pytest.fixture
def sample_products_df():
    n, rng = 50, np.random.default_rng(42)
    return pd.DataFrame({
        "product_id":        [f"PROD{i:04d}" for i in range(n)],
        "product_name":      [f"Product {i}" for i in range(n)],
        "category":          ["Electronics"] * n,
        "subcategory":       rng.choice(["Smartphones", "Laptops"], n),
        "brand":             [f"Brand{i % 10}" for i in range(n)],
        "base_price_2015":   rng.uniform(500, 50000, n),
        "is_prime_eligible": [True] * n,
    })


@pytest.fixture
def mock_engine():
    # engine.begin() — used by _upsert_dataframe (write transactions)
    # engine.connect() — used by train_anomaly_model (pd.read_sql read context)
    engine = MagicMock()
    mock_conn = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_conn)
    cm.__exit__  = MagicMock(return_value=False)
    engine.begin.return_value   = cm
    engine.connect.return_value = cm
    return engine


@pytest.fixture
def mock_anomaly_bundle(trained_anomaly_bundle):
    model, scaler = trained_anomaly_bundle
    return {
        "model":         model,
        "scaler":        scaler,
        "feature_names": ["final_amount_inr", "discount_pct", "delivery_days", "is_return"],
    }


@pytest.fixture
def mock_feast_features():
    from src.models.churn import ALL_FEATURES, NUMERIC_FEATURES
    row = {col: 1.0 for col in NUMERIC_FEATURES}
    row["rfm_segment"]        = "Loyal"        # in encoder categories
    row["preferred_category"] = "Electronics"  # in encoder categories
    row["customer_id"]        = "CUST0001"
    # Guard: verify all ALL_FEATURES are covered
    missing = set(ALL_FEATURES) - set(row.keys())
    assert not missing, f"mock_feast_features missing: {missing}"
    return pd.DataFrame([row])


@pytest.fixture
def mock_pricing_result():
    return {
        "subcategory":             "Smartphones",
        "current_price_inr":       20000.0,
        "current_demand_estimate": 150.0,
        "month":                   5,
        "is_festival_month":       False,
        "price_suggestions": [
            {"price_inr": 16000.0, "delta_pct": -20, "demand_estimate": 200.0, "revenue_estimate": 3200000.0},
            {"price_inr": 18000.0, "delta_pct": -10, "demand_estimate": 170.0, "revenue_estimate": 3060000.0},
            {"price_inr": 20000.0, "delta_pct":   0, "demand_estimate": 150.0, "revenue_estimate": 3000000.0},
            {"price_inr": 22000.0, "delta_pct":  10, "demand_estimate": 130.0, "revenue_estimate": 2860000.0},
            {"price_inr": 24000.0, "delta_pct":  20, "demand_estimate": 110.0, "revenue_estimate": 2640000.0},
        ],
    }


@pytest.fixture
def api_client_all_loaded(mock_anomaly_bundle):
    from starlette.testclient import TestClient
    from api.main import app
    with TestClient(app, raise_server_exceptions=True) as client:
        # Lifespan runs on __enter__ and sets all models to None (no artifacts).
        # Override with in-memory mocks immediately after.
        client.app.state.models = {
            "churn":          _make_churn_bundle(),
            "forecast":       {"smartphones": _make_prophet_mock()},
            "pricing":        _make_pricing_bundle(),
            "recommendation": {
                "rules":    pd.DataFrame(),
                "fallback": {"global_top": ["Laptops", "Tablets", "Smartphones"]},
            },
            "anomaly": mock_anomaly_bundle,
        }
        yield client


@pytest.fixture
def api_client_no_models():
    from starlette.testclient import TestClient
    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        client.app.state.models = {
            "churn": None, "forecast": {}, "pricing": None,
            "recommendation": None, "anomaly": None,
        }
        yield client


@pytest.fixture
def anomaly_mocks(tmp_path, monkeypatch, mock_engine):
    # anomaly.py line 13: from config import ARTIFACTS_DIR — bound in module namespace
    monkeypatch.setattr("src.models.anomaly.ARTIFACTS_DIR", tmp_path)
    (tmp_path / "models").mkdir()
    rng = np.random.default_rng(42)
    n = 200
    synthetic = pd.DataFrame({
        "transaction_id":   [f"T{i}" for i in range(n)],
        "final_amount_inr": rng.uniform(100, 50000, n),
        "mrp_inr":          rng.uniform(200, 60000, n),
        "delivery_days":    rng.uniform(0, 15, n),
        "is_return":        rng.choice([0, 1], n).astype(int),
    })
    return synthetic, tmp_path, mock_engine
