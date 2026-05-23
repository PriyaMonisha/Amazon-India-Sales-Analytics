from datetime import timedelta
from pathlib import Path

from feast import FeatureView, Field
from feast.infra.offline_stores.file_source import FileSource
from feast.types import Bool, Float64, Int64, String

from entities import customer_entity  # NO feast_repo. prefix — feast apply adds this dir to sys.path

_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

_training_source = FileSource(
    path=str(_PROCESSED_DIR / "customer_training_features.parquet"),
    timestamp_field="event_timestamp",
)

_prediction_source = FileSource(
    path=str(_PROCESSED_DIR / "customer_prediction_outputs.parquet"),
    timestamp_field="event_timestamp",
)

customer_training_fv = FeatureView(
    name="customer_training_features",
    entities=[customer_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="days_since_last_purchase",    dtype=Int64),
        Field(name="total_orders_90d",            dtype=Int64),
        Field(name="total_orders_all_time",       dtype=Int64),
        Field(name="avg_order_value_last_6m",     dtype=Float64),
        Field(name="avg_order_value_all_time",    dtype=Float64),
        Field(name="total_spend_all_time",        dtype=Float64),
        Field(name="is_prime_member",             dtype=Bool),
        Field(name="recency_score",               dtype=Int64),
        Field(name="frequency_score",             dtype=Int64),
        Field(name="monetary_score",              dtype=Int64),
        Field(name="rfm_segment",                 dtype=String),
        Field(name="preferred_category",          dtype=String),
        Field(name="unique_categories_purchased", dtype=Int64),
        Field(name="return_rate_historical",      dtype=Float64),
    ],
    source=_training_source,
    online=True,
)

customer_prediction_fv = FeatureView(
    name="customer_prediction_outputs",
    entities=[customer_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="churn_probability", dtype=Float64),
        Field(name="churn_label",       dtype=Int64),
        Field(name="clv_predicted",     dtype=Float64),
        Field(name="rfm_cluster",       dtype=Int64),
    ],
    source=_prediction_source,
    online=True,
)
