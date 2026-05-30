"""
DDL for A/B testing tables.

ab_predictions: every churn prediction made during an A/B experiment.
ab_outcomes:    ground-truth churn labels (populated by materialize_ab_outcomes.py).
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


_AB_DDL = text("""
CREATE TABLE IF NOT EXISTS ab_predictions (
    id            SERIAL PRIMARY KEY,
    customer_id   TEXT        NOT NULL,
    model_version TEXT        NOT NULL CHECK (model_version IN ('champion', 'challenger')),
    churn_prob    FLOAT       NOT NULL,
    predicted_at  TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);
CREATE INDEX IF NOT EXISTS idx_ab_cust_time ON ab_predictions (customer_id, predicted_at);

CREATE TABLE IF NOT EXISTS ab_outcomes (
    customer_id  TEXT    PRIMARY KEY,
    churned      BOOLEAN NOT NULL,
    outcome_date DATE    NOT NULL
);
""")


def create_ab_tables(engine: Engine) -> None:
    """Idempotent table creation. Safe to call multiple times."""
    with engine.begin() as conn:
        conn.execute(_AB_DDL)
