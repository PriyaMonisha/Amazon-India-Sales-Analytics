import utils  # noqa: F401 — sets sys.path to /opt/airflow; must be first import

import pendulum
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.exceptions import AirflowException

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="etl_pipeline",
    default_args=default_args,
    start_date=pendulum.datetime(2025, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["etl", "section-8"],
    doc_md="""
## ETL Pipeline

Extracts raw CSVs → transforms + validates → loads to PostgreSQL.
On success, triggers the `feature_engineering` DAG via TriggerDagRunOperator.

**Schedule:** @weekly (also manually triggerable)
""",
) as dag:

    def _extract(**context):
        import config
        from src.etl.extract import load_raw_csvs, load_product_catalog

        ti = context["ti"]
        raw_sales = load_raw_csvs(config.RAW_DATA_DIR)
        raw_products = load_product_catalog(config.RAW_DATA_DIR)

        sales_path = config.PROCESSED_DATA_DIR / "airflow_raw_sales.parquet"
        products_path = config.PROCESSED_DATA_DIR / "airflow_raw_products.parquet"
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        raw_sales.to_parquet(sales_path, index=False)
        raw_products.to_parquet(products_path, index=False)

        ti.xcom_push(key="raw_sales_path", value=str(sales_path))
        ti.xcom_push(key="raw_products_path", value=str(products_path))
        print(f"Extracted {len(raw_sales):,} sales rows, {len(raw_products):,} product rows")

    def _transform(**context):
        import pandas as pd
        import config
        from src.etl.transform import clean_sales, clean_products, build_cleaning_log

        ti = context["ti"]
        sales_path = ti.xcom_pull(task_ids="extract", key="raw_sales_path")
        products_path = ti.xcom_pull(task_ids="extract", key="raw_products_path")

        raw_sales = pd.read_parquet(sales_path)
        raw_products = pd.read_parquet(products_path)

        clean_sales_df, cleaning_log = clean_sales(raw_sales)
        clean_products_df = clean_products(raw_products)
        build_cleaning_log(cleaning_log, config.ARTIFACTS_DIR)

        clean_sales_path = config.PROCESSED_DATA_DIR / "airflow_clean_sales.parquet"
        clean_products_path = config.PROCESSED_DATA_DIR / "airflow_clean_products.parquet"

        clean_sales_df.to_parquet(clean_sales_path, index=False)
        clean_products_df.to_parquet(clean_products_path, index=False)

        ti.xcom_push(key="clean_sales_path", value=str(clean_sales_path))
        ti.xcom_push(key="clean_products_path", value=str(clean_products_path))
        print(f"Cleaned: {len(clean_sales_df):,} sales rows, {len(clean_products_df):,} product rows")

    def _validate(**context):
        import pandas as pd
        from src.validation.expectations import validate_sales

        ti = context["ti"]
        clean_sales_path = ti.xcom_pull(task_ids="transform", key="clean_sales_path")
        df = pd.read_parquet(clean_sales_path)

        passed = validate_sales(df)
        if not passed:
            raise AirflowException("Pandera validation failed — check logs for schema violations")
        print(f"Validation passed for {len(df):,} rows")

    def _load(**context):
        import pandas as pd
        import config
        from src.etl.load import get_engine, create_tables, load_dimensions, load_facts

        ti = context["ti"]
        clean_sales_path = ti.xcom_pull(task_ids="transform", key="clean_sales_path")
        clean_products_path = ti.xcom_pull(task_ids="transform", key="clean_products_path")

        df_sales = pd.read_parquet(clean_sales_path)
        df_products = pd.read_parquet(clean_products_path)

        engine = get_engine(config.DB_URL)
        create_tables(engine)

        df_customers = df_sales[["customer_id", "customer_name", "customer_city",
                                  "customer_state", "is_prime_member"]].drop_duplicates("customer_id")
        load_dimensions(engine, df_customers, df_products)
        load_facts(engine, df_sales)
        print(f"Loaded {len(df_sales):,} fact rows to PostgreSQL")

    extract = PythonOperator(task_id="extract", python_callable=_extract)
    transform = PythonOperator(task_id="transform", python_callable=_transform)
    validate = PythonOperator(task_id="validate", python_callable=_validate)
    load = PythonOperator(task_id="load", python_callable=_load)
    trigger_feature_eng = TriggerDagRunOperator(
        task_id="trigger_feature_engineering",
        trigger_dag_id="feature_engineering",
        wait_for_completion=False,
    )

    extract >> transform >> validate >> load >> trigger_feature_eng
