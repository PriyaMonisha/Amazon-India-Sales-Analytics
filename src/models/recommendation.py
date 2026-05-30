import json
import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config import ARTIFACTS_DIR, FAST_MODE, RANDOM_STATE, SAMPLE_ROWS
from src.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

# Progressive min_support thresholds — first that yields rules is used
_MIN_SUPPORT_CANDIDATES = [0.005, 0.002, 0.001]

# SQL: order-level transaction data (transaction_id × subcategory)
_BASKET_QUERY = text("""
SELECT
    ft.transaction_id,
    dp.subcategory
FROM fact_transactions ft
JOIN dim_products dp ON ft.product_id = dp.product_id
WHERE ft.transaction_id IS NOT NULL
  AND dp.subcategory IS NOT NULL
""")

# SQL: global subcategory popularity (order count)
_POPULARITY_QUERY = text("""
SELECT
    dp.subcategory,
    COUNT(DISTINCT ft.transaction_id) AS order_count
FROM fact_transactions ft
JOIN dim_products dp ON ft.product_id = dp.product_id
WHERE dp.subcategory IS NOT NULL
GROUP BY dp.subcategory
ORDER BY order_count DESC
""")


def _build_popularity_fallback(popularity_df: pd.DataFrame) -> dict[str, Any]:
    """Builds global popularity ranking for cold-start fallback."""
    top = popularity_df.head(50)
    return {
        "global_top": top["subcategory"].tolist(),
        "order_counts": top["order_count"].tolist(),
    }


def train_recommendation_model(engine: Engine) -> pd.DataFrame:
    """
    Trains FP-Growth association rules on ORDER-level baskets.

    Uses progressive min_support [0.005, 0.002, 0.001] — first threshold that
    yields rules. Falls back to popularity ranking if no rules at 0.001.

    FAST_MODE: samples 100k transactions before basket construction.

    Saves:
        artifacts/models/association_rules.parquet
        artifacts/models/popularity_fallback.json

    Returns the association_rules DataFrame.
    """
    models_dir = ARTIFACTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load basket data
    logger.info("Loading basket data from PostgreSQL...")
    with engine.connect() as conn:
        df      = pd.read_sql(_BASKET_QUERY, conn)
        pop_df  = pd.read_sql(_POPULARITY_QUERY, conn)

    if df.empty:
        raise RuntimeError("No basket data returned — check ETL pipeline")

    logger.info("Loaded %d transaction-subcategory rows", len(df))

    # FAST_MODE: sample transactions (not rows — sample unique transaction_ids)
    if FAST_MODE and SAMPLE_ROWS is not None:
        max_txns = 100_000
        unique_txns = df["transaction_id"].unique()
        if len(unique_txns) > max_txns:
            rng = np.random.RandomState(RANDOM_STATE)
            sampled_txns = rng.choice(unique_txns, size=max_txns, replace=False)
            df = df[df["transaction_id"].isin(sampled_txns)]
            logger.info("FAST_MODE: sampled %d transactions", max_txns)

    # Build order-level boolean basket matrix
    # Rows = transaction_id, Columns = subcategory, Values = True/False
    basket = (
        df.groupby(["transaction_id", "subcategory"])
        .size()
        .unstack(fill_value=0)
        .astype(bool)   # mlxtend requires bool dtype
    )
    logger.info("Basket matrix: %d orders × %d subcategories", *basket.shape)

    # Popularity fallback (computed before FP-Growth attempt)
    fallback = _build_popularity_fallback(pop_df)
    fallback_path = models_dir / "popularity_fallback.json"
    with open(fallback_path, "w") as f:
        json.dump(fallback, f, indent=2)

    # Progressive min_support search
    frequent_items: pd.DataFrame = pd.DataFrame()
    rules: pd.DataFrame = pd.DataFrame()
    min_support_used: float | None = None

    for min_sup in _MIN_SUPPORT_CANDIDATES:
        logger.info("Trying min_support=%.3f...", min_sup)
        try:
            frequent_items = fpgrowth(basket, min_support=min_sup, use_colnames=True)
        except Exception as exc:
            logger.warning("FP-Growth failed at min_support=%.3f: %s", min_sup, exc)
            continue

        if frequent_items.empty:
            logger.info("  No frequent itemsets at %.3f", min_sup)
            continue

        rules = association_rules(
            frequent_items, metric="lift", min_threshold=1.0
        )
        if not rules.empty:
            min_support_used = min_sup
            logger.info(
                "  Found %d rules at min_support=%.3f", len(rules), min_sup
            )
            break
        logger.info("  Frequent itemsets found but no rules with lift >= 1.0")

    # Prepare rules DataFrame (or empty if fallback only)
    if rules.empty:
        logger.warning(
            "No association rules found at any min_support — serving popularity fallback only"
        )
        rules = pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )
    else:
        # Convert frozensets to sorted strings for Parquet serialization
        rules = rules.copy()
        rules["antecedents"] = rules["antecedents"].apply(lambda x: sorted(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: sorted(list(x)))
        # Sort by lift descending for fast lookup
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    # Save to disk FIRST, then MLflow
    rules_path = models_dir / "association_rules.parquet"
    rules.to_parquet(rules_path, index=False)

    setup_mlflow("amazon_recommendations")
    with mlflow.start_run(run_name="fp_growth_recommendations") as run:
        mlflow.log_params({
            "min_support_used":         min_support_used if min_support_used else "none",
            "candidates_tried":         str(_MIN_SUPPORT_CANDIDATES),
            "fast_mode":                FAST_MODE,
            "basket_orders":            basket.shape[0],
            "basket_subcategories":     basket.shape[1],
        })
        mlflow.log_metrics({
            "n_frequent_itemsets": len(frequent_items),
            "n_rules":             len(rules),
            "avg_lift":            float(rules["lift"].mean()) if not rules.empty else 0.0,
            "max_lift":            float(rules["lift"].max()) if not rules.empty else 0.0,
        })
        mlflow.log_artifact(str(rules_path))
        mlflow.log_artifact(str(fallback_path))
        logger.info(
            "Recommendation model saved | %d rules | run_id=%s",
            len(rules), run.info.run_id,
        )

    return rules


def load_recommendation_model() -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Loads association rules and popularity fallback from disk.

    Returns (rules_df, fallback_dict).
    """
    models_dir = ARTIFACTS_DIR / "models"

    rules_path    = models_dir / "association_rules.parquet"
    fallback_path = models_dir / "popularity_fallback.json"

    rules: pd.DataFrame = pd.DataFrame()
    if rules_path.exists():
        rules = pd.read_parquet(rules_path)

    fallback: dict[str, Any] = {}
    if fallback_path.exists():
        with open(fallback_path) as f:
            fallback = json.load(f)

    logger.info(
        "Recommendation model loaded: %d rules, %d global popular subcategories",
        len(rules), len(fallback.get("global_top", [])),
    )
    return rules, fallback


def get_recommendations(
    subcategory: str,
    top_n: int = 5,
    rules: pd.DataFrame | None = None,
    fallback: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Returns up to top_n product recommendations for a given subcategory.

    Cold-start tiers:
        1. Association rule consequents where antecedent contains subcategory
        2. Global popularity fallback

    Pass pre-loaded rules + fallback for FastAPI (avoids re-loading per request).
    """
    if rules is None or fallback is None:
        rules, fallback = load_recommendation_model()

    recs: list[dict[str, Any]] = []

    # Tier 1: personalized rules
    if not rules.empty and "antecedents" in rules.columns:
        matched = rules[
            rules["antecedents"].apply(
                lambda ants: subcategory in (ants if isinstance(ants, list) else list(ants))
            )
        ]
        for _, row in matched.head(top_n).iterrows():
            consequents = row["consequents"] if isinstance(row["consequents"], list) else list(row["consequents"])
            for cons in consequents:
                if cons != subcategory and len(recs) < top_n:
                    recs.append({
                        "subcategory": cons,
                        "confidence":  round(float(row["confidence"]), 4),
                        "lift":        round(float(row["lift"]), 4),
                        "source":      "association_rules",
                    })

    # Tier 2: popularity fallback (fill remaining slots)
    global_top: list[str] = fallback.get("global_top", [])
    for subcat in global_top:
        if len(recs) >= top_n:
            break
        if subcat != subcategory and not any(r["subcategory"] == subcat for r in recs):
            recs.append({
                "subcategory": subcat,
                "confidence":  None,
                "lift":        None,
                "source":      "popularity_fallback",
            })

    return recs[:top_n]
