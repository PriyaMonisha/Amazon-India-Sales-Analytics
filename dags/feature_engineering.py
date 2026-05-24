import utils  # noqa: F401 — sets sys.path to /opt/airflow; must be first import

import pendulum
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="feature_engineering",
    default_args=default_args,
    start_date=pendulum.datetime(2025, 1, 1),
    schedule=None,  # triggered by etl_pipeline via TriggerDagRunOperator
    catchup=False,
    tags=["features", "feast", "section-8"],
    doc_md="""
## Feature Engineering

Computes RFM + behavioural customer features from PostgreSQL, then materializes
them to the Feast online store (Redis) for low-latency serving.

**Triggered by:** `etl_pipeline` (TriggerDagRunOperator)
**Triggers:** `model_training`
""",
) as dag:

    def _compute_features(**context):
        import config
        from src.etl.load import get_engine
        from src.features.compute import compute_customer_features

        engine = get_engine(config.DB_URL)
        reference_date = datetime.utcnow().date()
        df = compute_customer_features(engine, reference_date=reference_date)
        print(f"Computed features for {len(df):,} customers (ref_date={reference_date})")

    def _feast_materialize(**context):
        # Full rematerialization on every run — acceptable for portfolio scale.
        # Production: use pendulum.now("UTC").subtract(days=7) as start_date after first run.
        from src.features.feature_store import materialize_customer_features

        start_date = pendulum.datetime(2015, 1, 1, timezone="UTC")
        end_date = pendulum.now("UTC")
        materialize_customer_features(start_date=start_date, end_date=end_date)
        print(f"Feast materialization complete: {start_date.date()} → {end_date.date()}")

    compute_features = PythonOperator(
        task_id="compute_features",
        python_callable=_compute_features,
    )
    feast_materialize = PythonOperator(
        task_id="feast_materialize",
        python_callable=_feast_materialize,
    )
    trigger_model_train = TriggerDagRunOperator(
        task_id="trigger_model_training",
        trigger_dag_id="model_training",
        wait_for_completion=False,
    )

    compute_features >> feast_materialize >> trigger_model_train
