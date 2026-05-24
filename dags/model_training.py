import utils  # noqa: F401 — sets sys.path to /opt/airflow; must be first import

import pendulum
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    # Longer delay: one retry handles transient DB connection failures.
    # Model logic failures surface on the retry without waiting.
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="model_training",
    default_args=default_args,
    start_date=pendulum.datetime(2025, 1, 1),
    schedule=None,  # triggered by feature_engineering via TriggerDagRunOperator
    catchup=False,
    tags=["ml", "training", "section-8"],
    doc_md="""
## Model Training

Trains all 5 ML models in parallel, then triggers the `model_serving_check` DAG.

**Triggered by:** `feature_engineering`
**Triggers:** `model_serving_check`

**Important — FastAPI model reload:**
FastAPI loads models at startup via the lifespan hook. After training writes new
artifacts, FastAPI still serves the startup models until the container restarts.
The serving check is only valid after `docker compose restart fastapi`.
In production, add a /admin/reload endpoint or use a model registry with push
notifications instead of file-based artifact loading.
""",
) as dag:

    def _train_churn(**context):
        import config
        from src.etl.load import get_engine
        from src.models.churn import train_churn_model

        engine = get_engine(config.DB_URL)
        bundle = train_churn_model(engine)
        print(f"Churn model trained — threshold={bundle['threshold']:.3f}, "
              f"churn_rate_test={bundle['churn_rate_test']:.3f}")

    def _train_forecast(**context):
        import config
        from src.etl.load import get_engine
        from src.models.forecasting import train_forecast_models

        engine = get_engine(config.DB_URL)
        wmape_scores = train_forecast_models(engine)
        n = len(wmape_scores)
        avg_wmape = sum(wmape_scores.values()) / n if n else 0
        print(f"Forecast models trained — {n} subcategories, avg WMAPE={avg_wmape:.3f}")

    def _train_pricing(**context):
        import config
        from src.etl.load import get_engine
        from src.models.pricing import train_pricing_model

        engine = get_engine(config.DB_URL)
        result = train_pricing_model(engine)
        print(f"Pricing model trained — r2={result.get('r2', 'n/a')}")

    def _train_recommendation(**context):
        import config
        from src.etl.load import get_engine
        from src.models.recommendation import train_recommendation_model

        engine = get_engine(config.DB_URL)
        df = train_recommendation_model(engine)
        print(f"Recommendation model trained — {len(df):,} association rules")

    def _train_anomaly(**context):
        import config
        from src.etl.load import get_engine
        from src.models.anomaly import train_anomaly_model

        engine = get_engine(config.DB_URL)
        result = train_anomaly_model(engine)
        print(f"Anomaly model trained — roc_auc={result.get('roc_auc', 'n/a')}")

    train_churn = PythonOperator(task_id="train_churn", python_callable=_train_churn)
    train_forecast = PythonOperator(task_id="train_forecast", python_callable=_train_forecast)
    train_pricing = PythonOperator(task_id="train_pricing", python_callable=_train_pricing)
    train_recommendation = PythonOperator(task_id="train_recommendation", python_callable=_train_recommendation)
    train_anomaly = PythonOperator(task_id="train_anomaly", python_callable=_train_anomaly)

    trigger_serving_check = TriggerDagRunOperator(
        task_id="trigger_model_serving_check",
        trigger_dag_id="model_serving_check",
        wait_for_completion=False,
    )

    # All 5 training tasks run in parallel; trigger serving check only after all pass
    [train_churn, train_forecast, train_pricing, train_recommendation, train_anomaly] >> trigger_serving_check
