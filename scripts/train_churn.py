"""
Standalone churn training script.

Invoked by the Airflow model_retraining DAG via BashOperator to keep Airflow
isolated from the application's runtime context (no shared in-memory state).

Usage:
    python scripts/train_churn.py

Environment variables required:
    AMAZON_DB_URL   — PostgreSQL connection string
    FAST_MODE       — (optional) "true" for reduced sample / trials
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure project root is on the path when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    db_url = os.environ.get("AMAZON_DB_URL")
    if not db_url:
        logger.error("AMAZON_DB_URL environment variable is not set")
        sys.exit(1)

    from src.models.churn import train_churn_model
    engine = create_engine(db_url)
    logger.info("Starting churn model training...")
    bundle = train_churn_model(engine)
    logger.info(
        "Training complete — churn_rate_test=%.1f%%",
        bundle["churn_rate_test"] * 100,
    )


if __name__ == "__main__":
    main()
