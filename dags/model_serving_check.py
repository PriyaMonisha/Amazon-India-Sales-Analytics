import utils  # noqa: F401 — sets sys.path to /opt/airflow; must be first import

import pendulum
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException

FASTAPI_BASE = "http://fastapi:8000"

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def _fastapi_get(url: str) -> dict:
    import requests
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise AirflowException(f"FastAPI unreachable at {FASTAPI_BASE} — is the container running?")
    except requests.exceptions.HTTPError as e:
        raise AirflowException(f"FastAPI call failed: {e.response.status_code} — {e.response.text}")
    return resp.json()


def _fastapi_post(url: str, payload: dict) -> dict:
    import requests
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise AirflowException(f"FastAPI unreachable at {FASTAPI_BASE} — is the container running?")
    except requests.exceptions.HTTPError as e:
        raise AirflowException(f"FastAPI call failed: {e.response.status_code} — {e.response.text}")
    return resp.json()


with DAG(
    dag_id="model_serving_check",
    default_args=default_args,
    start_date=pendulum.datetime(2025, 1, 1),
    schedule=None,  # triggered by model_training via TriggerDagRunOperator
    catchup=False,
    tags=["serving", "health", "section-8"],
    doc_md="""
## Model Serving Check

Verifies that FastAPI is up and all 5 model endpoints respond correctly,
including basic response content assertions to catch silently broken models.

**Triggered by:** `model_training`

**Note on model reload:**
FastAPI loads models at startup. After training writes new artifacts, FastAPI
still serves startup models until `docker compose restart fastapi`. This check
is therefore most useful after a full-stack restart post-training.
In production: add /admin/reload or use a model registry with push notifications.
""",
) as dag:

    def _check_health(**context):
        data = _fastapi_get(f"{FASTAPI_BASE}/health")
        if data.get("status") != "ok":
            raise AirflowException(f"Health check failed: status={data.get('status')}")
        models_loaded = data.get("models_loaded", {})
        expected_keys = {"churn", "forecast", "pricing", "recommendation", "anomaly"}
        missing = expected_keys - set(models_loaded.keys())
        if missing:
            raise AirflowException(f"models_loaded missing keys: {missing}")
        print(f"Health OK — models_loaded: {models_loaded}")

    def _check_churn(**context):
        data = _fastapi_get(f"{FASTAPI_BASE}/predict/churn/CUST001")
        prob = data.get("churn_probability")
        if not (isinstance(prob, float) and 0.0 <= prob <= 1.0):
            raise AirflowException(f"churn_probability out of range: {prob!r}")
        if "churn_predicted" not in data:
            raise AirflowException(f"churn_predicted key missing from response: {data}")
        print(f"Churn check OK — probability={prob:.3f}, predicted={data['churn_predicted']}")

    def _check_forecast(**context):
        import json
        import config

        slugs_path = config.ARTIFACTS_DIR / "models" / "forecast_slugs.json"
        if not slugs_path.exists():
            raise AirflowException(f"forecast_slugs.json not found at {slugs_path} — run model_training first")
        slugs = json.loads(slugs_path.read_text())
        first_slug = next(iter(slugs))

        data = _fastapi_get(f"{FASTAPI_BASE}/predict/forecast/{first_slug}?periods=1")
        forecast = data.get("forecast", [])
        if len(forecast) != 1:
            raise AirflowException(f"Expected 1 forecast point for periods=1, got {len(forecast)}")
        print(f"Forecast check OK — subcategory={first_slug}, points={len(forecast)}")

    def _check_pricing(**context):
        data = _fastapi_get(f"{FASTAPI_BASE}/predict/pricing/Smartphones?current_price=15000")
        suggestions = data.get("price_suggestions", [])
        if len(suggestions) != 5:
            raise AirflowException(f"Expected 5 price suggestions, got {len(suggestions)}: {suggestions}")
        print(f"Pricing check OK — {len(suggestions)} price suggestions")

    def _check_recommendation(**context):
        data = _fastapi_get(f"{FASTAPI_BASE}/predict/recommend/Smartphones?top_n=3")
        recs = data.get("recommendations", [])
        if len(recs) == 0:
            raise AirflowException("Recommendation endpoint returned empty list")
        missing_fields = [r for r in recs if "subcategory" not in r or "source" not in r]
        if missing_fields:
            raise AirflowException(f"Malformed recommendation items (missing subcategory/source): {missing_fields[0]}")
        print(f"Recommendation check OK — {len(recs)} items, sources={[r['source'] for r in recs]}")

    def _check_anomaly(**context):
        payload = {
            "final_amount_inr": 12000.0,
            "mrp_inr": 15000.0,
            "delivery_days": 3.0,
            "is_return": 0,
        }
        data = _fastapi_post(f"{FASTAPI_BASE}/predict/anomaly", payload)
        if "is_anomaly" not in data:
            raise AirflowException(f"is_anomaly key missing from anomaly response: {data}")
        if not isinstance(data.get("anomaly_score"), float):
            raise AirflowException(f"anomaly_score is not a float: {data.get('anomaly_score')!r}")
        print(f"Anomaly check OK — is_anomaly={data['is_anomaly']}, score={data['anomaly_score']:.4f}")

    check_health = PythonOperator(task_id="check_health", python_callable=_check_health)
    check_churn = PythonOperator(task_id="check_churn", python_callable=_check_churn)
    check_forecast = PythonOperator(task_id="check_forecast", python_callable=_check_forecast)
    check_pricing = PythonOperator(task_id="check_pricing", python_callable=_check_pricing)
    check_recommendation = PythonOperator(task_id="check_recommendation", python_callable=_check_recommendation)
    check_anomaly = PythonOperator(task_id="check_anomaly", python_callable=_check_anomaly)

    check_health >> [check_churn, check_forecast, check_pricing, check_recommendation, check_anomaly]
