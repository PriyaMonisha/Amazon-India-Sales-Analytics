import logging
import os
from datetime import datetime

import pandas as pd
from feast import FeatureStore

import config

logger = logging.getLogger(__name__)

CUSTOMER_TRAINING_FEATURE_REFS: list[str] = [
    "customer_training_features:days_since_last_purchase",
    "customer_training_features:total_orders_90d",
    "customer_training_features:total_orders_all_time",
    "customer_training_features:avg_order_value_last_6m",
    "customer_training_features:avg_order_value_all_time",
    "customer_training_features:total_spend_all_time",
    "customer_training_features:is_prime_member",
    "customer_training_features:recency_score",
    "customer_training_features:frequency_score",
    "customer_training_features:monetary_score",
    "customer_training_features:rfm_segment",
    "customer_training_features:preferred_category",
    "customer_training_features:unique_categories_purchased",
    "customer_training_features:return_rate_historical",
]


def get_feature_store() -> FeatureStore:
    """Returns configured FeatureStore with Redis overridden from config.py."""
    os.environ["REDIS_CONNECTION_STRING"] = f"{config.REDIS_HOST}:{config.REDIS_PORT}"
    return FeatureStore(repo_path=str(config.FEAST_REPO_PATH))


def materialize_customer_features(start_date: datetime, end_date: datetime) -> None:
    """
    Materializes customer_training_features to Redis.

    Explicitly scoped to customer_training_features only —
    customer_prediction_outputs Parquet doesn't exist until Section 5 inference runs.
    """
    try:
        store = get_feature_store()
        store.materialize(
            start_date=start_date,
            end_date=end_date,
            feature_views=["customer_training_features"],
        )
        logger.info(
            "Materialized customer_training_features: %s → %s", start_date, end_date
        )
    except Exception:
        logger.exception("materialize_customer_features failed")


def get_online_features(customer_ids: list[str]) -> pd.DataFrame:
    """
    Retrieves customer training features from Redis for inference.

    Returns empty DataFrame on failure — caller must check len(df) > 0.
    """
    try:
        store = get_feature_store()
        result = store.get_online_features(
            features=CUSTOMER_TRAINING_FEATURE_REFS,
            entity_rows=[{"customer_id": cid} for cid in customer_ids],
        )
        return result.to_df()
    except Exception:
        logger.exception("get_online_features failed")
        return pd.DataFrame()


def get_training_dataset(
    entity_df: pd.DataFrame,
    feature_refs: list[str] | None = None,
) -> pd.DataFrame:
    """
    Point-in-time correct feature retrieval for model training.

    entity_df must have: customer_id (str), event_timestamp (datetime).
    Automatically localizes event_timestamp to UTC if tz-naive.

    Returns empty DataFrame on failure — caller must check len(df) > 0.
    FileNotFoundError → compute_customer_features() hasn't been run yet.
    """
    if "event_timestamp" in entity_df.columns:
        if entity_df["event_timestamp"].dt.tz is None:
            entity_df = entity_df.copy()
            entity_df["event_timestamp"] = entity_df["event_timestamp"].dt.tz_localize(
                "UTC"
            )
    try:
        store = get_feature_store()
        refs = feature_refs or CUSTOMER_TRAINING_FEATURE_REFS
        return store.get_historical_features(
            entity_df=entity_df, features=refs
        ).to_df()
    except FileNotFoundError as e:
        logger.error(
            "Feature Parquet not found — run compute_customer_features() first: %s", e
        )
        return pd.DataFrame()
    except Exception:
        logger.exception("get_training_dataset failed")
        return pd.DataFrame()
