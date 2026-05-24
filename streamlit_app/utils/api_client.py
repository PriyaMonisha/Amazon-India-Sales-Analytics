from __future__ import annotations

import requests

import config

_BASE = config.FASTAPI_URL
_TIMEOUT = 10


def _get(path: str, params: dict | None = None) -> dict:
    try:
        r = requests.get(f"{_BASE}{path}", params=params, timeout=_TIMEOUT)
        if r.ok:
            return r.json()
        return {"error": r.json().get("detail", f"HTTP {r.status_code}")}
    except requests.exceptions.ConnectionError:
        return {"error": "FastAPI not reachable — start the API server first"}
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, body: dict) -> dict:
    try:
        r = requests.post(f"{_BASE}{path}", json=body, timeout=_TIMEOUT)
        if r.ok:
            return r.json()
        return {"error": r.json().get("detail", f"HTTP {r.status_code}")}
    except requests.exceptions.ConnectionError:
        return {"error": "FastAPI not reachable — start the API server first"}
    except Exception as e:
        return {"error": str(e)}


def get_health() -> dict:
    return _get("/health")


def get_churn(customer_id: str) -> dict:
    return _get(f"/predict/churn/{customer_id}")


def get_churn_explain(customer_id: str) -> dict:
    return _get(f"/predict/churn/{customer_id}/explain")


def get_forecast(subcategory: str, periods: int = 6) -> dict:
    return _get(f"/predict/forecast/{subcategory}", {"periods": periods})


def get_pricing(subcategory: str, current_price: float) -> dict:
    return _get(f"/predict/pricing/{subcategory}", {"current_price": current_price})


def get_recommendations(subcategory: str, top_n: int = 10) -> dict:
    return _get(f"/predict/recommend/{subcategory}", {"top_n": top_n})


def post_anomaly(
    final_amount_inr: float,
    mrp_inr: float,
    delivery_days: float,
    is_return: int = 0,
) -> dict:
    return _post("/predict/anomaly", {
        "final_amount_inr": final_amount_inr,
        "mrp_inr": mrp_inr,
        "delivery_days": delivery_days,
        "is_return": is_return,
    })
