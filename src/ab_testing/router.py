"""
Deterministic A/B routing for churn model champion vs challenger.

Hash-based routing ensures the same customer always receives predictions from
the same model version throughout an experiment. Random routing would break
outcome attribution and produce biased AUC comparisons.

Usage:
    version = choose_version("CUST001", challenger_pct=0.10)
    # → "champion" or "challenger", always the same for the same customer_id
"""
from __future__ import annotations

import hashlib
from typing import Literal

ModelVersion = Literal["champion", "challenger"]


def choose_version(customer_id: str, challenger_pct: float) -> ModelVersion:
    """
    Deterministically assign a customer to champion or challenger.

    Converts customer_id to a SHA-256 hash, takes the last 8 hex digits as an
    integer, maps to bucket [0, 100). Customers with bucket < challenger_pct*100
    receive the challenger model.

    Args:
        customer_id:     Unique customer identifier (any string).
        challenger_pct:  Fraction of traffic routed to challenger (0.0 to 1.0).

    Returns:
        "challenger" or "champion".
    """
    if not (0.0 <= challenger_pct <= 1.0):
        raise ValueError(f"challenger_pct must be in [0, 1], got {challenger_pct}")
    digest = hashlib.sha256(customer_id.encode()).hexdigest()
    bucket = int(digest[-8:], 16) % 100   # stable 32-bit bucket, collision-resistant
    return "challenger" if bucket < int(challenger_pct * 100) else "champion"
