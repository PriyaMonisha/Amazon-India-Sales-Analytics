"""
Nightly A/B outcome materialization script.

For every customer in ab_predictions whose prediction was made >= 90 days ago,
determines whether they churned by checking if they made any purchase in the
90-day observation window after the prediction date.

churned = True  if no purchase in the window (customer did not return)
churned = False if at least one purchase in the window (customer returned)

Upserts into ab_outcomes so each customer has one canonical label.

Usage:
    python scripts/materialize_ab_outcomes.py

Environment:
    AMAZON_DB_URL — PostgreSQL connection string (required)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_MATERIALIZE_SQL = text("""
INSERT INTO ab_outcomes (customer_id, churned, outcome_date)
SELECT
    p.customer_id,
    -- churned = TRUE if no purchase in 90-day window after prediction
    (COUNT(ft.transaction_id) = 0)::boolean AS churned,
    CURRENT_DATE                            AS outcome_date
FROM (
    SELECT DISTINCT customer_id, MIN(predicted_at) AS first_prediction_at
    FROM ab_predictions
    WHERE predicted_at <= NOW() - INTERVAL '90 days'
    GROUP BY customer_id
) p
LEFT JOIN fact_transactions ft
    ON ft.customer_id = p.customer_id
    AND ft.order_date >= p.first_prediction_at::date
    AND ft.order_date <  p.first_prediction_at::date + INTERVAL '90 days'
WHERE p.customer_id NOT IN (SELECT customer_id FROM ab_outcomes)
GROUP BY p.customer_id
ON CONFLICT (customer_id) DO UPDATE
    SET churned      = EXCLUDED.churned,
        outcome_date = EXCLUDED.outcome_date
""")


def main() -> None:
    db_url = os.environ.get("AMAZON_DB_URL")
    if not db_url:
        logger.error("AMAZON_DB_URL not set")
        sys.exit(1)

    engine = create_engine(db_url)
    with engine.begin() as conn:
        result = conn.execute(_MATERIALIZE_SQL)
        logger.info("Materialized %d ab_outcomes rows", result.rowcount)


if __name__ == "__main__":
    main()
