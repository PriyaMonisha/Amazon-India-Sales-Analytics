import utils  # noqa: F401 — sets sys.path to /opt/airflow; must be first import

import logging
import pendulum
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException

FASTAPI_BASE = "http://fastapi:8000"

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="drift_monitoring",
    default_args=default_args,
    start_date=pendulum.datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["monitoring", "drift", "section-8"],
    doc_md="""
## Drift Monitoring

Calls FastAPI's `/monitor/drift/run` endpoint daily to drain the prediction buffer
and run Evidently data drift detection. FastAPI's process updates its Prometheus
gauges, which Prometheus then scrapes and Grafana displays.

Calling `compute_churn_drift()` directly from Airflow would update a process-local
Prometheus registry that Prometheus never scrapes — hence the HTTP call approach.

**Note:** Returns `status: skipped` when the prediction buffer is empty (e.g. at 3am
with no recent traffic). This is normal and logged as a warning, not an error.
""",
) as dag:

    def _run_drift_check(**context):
        import requests

        try:
            resp = requests.post(f"{FASTAPI_BASE}/monitor/drift/run", timeout=10)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise AirflowException(
                f"FastAPI unreachable at {FASTAPI_BASE} — is the container running?"
            )
        except requests.exceptions.HTTPError as e:
            raise AirflowException(
                f"Drift check call failed: {e.response.status_code} — {e.response.text}"
            )

        result = resp.json()
        if result.get("status") == "skipped":
            logger.warning("Drift check skipped: %s", result.get("reason"))
        else:
            logger.info(
                "Drift check complete: dataset_drift=%s, columns_drifted=%s/%s",
                result.get("dataset_drift"),
                result.get("columns_drifted"),
                result.get("columns_checked"),
            )
        print(f"Drift check result: {result}")

    run_drift_check = PythonOperator(
        task_id="run_drift_check",
        python_callable=_run_drift_check,
    )
