"""
Lightweight JSON-manifest model registry.

Atomic writes via filelock + tempfile prevent JSON corruption under concurrent
Airflow task execution or API restarts.

Registry layout:
    artifacts/registry.json          — master manifest (list of ModelRecord dicts)
    artifacts/registry.lock          — lock file (never committed)
    artifacts/registry/
        promotion_log.json           — append-only audit trail

Versioned artifacts live in:
    artifacts/models/<model_name>/<version>/
"""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from config import ARTIFACTS_DIR

REGISTRY_PATH = ARTIFACTS_DIR / "registry.json"
REGISTRY_LOCK = ARTIFACTS_DIR / "registry.lock"
PROMOTION_LOG = ARTIFACTS_DIR / "registry" / "promotion_log.json"


@dataclass
class ModelRecord:
    version: str                      # e.g. "20260530_143212"
    model_name: str                   # "churn" | "pricing" | "forecasting" | ...
    artifact_dir: str                 # path relative to ARTIFACTS_DIR
    status: str                       # "candidate" | "production" | "retired"
    trained_at: str                   # UTC ISO-8601 timestamp
    roc_auc: float | None = None
    r2: float | None = None
    threshold: float | None = None
    feature_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # forward-compat extension point


# ---------------------------------------------------------------------------
# Internal I/O helpers
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: Any) -> None:
    """Write JSON atomically — no partial writes on crash or concurrent access."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, suffix=".tmp", encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = Path(tmp.name)
    shutil.move(str(tmp_path), path)


def _read_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register(record: ModelRecord) -> None:
    """Add or update a model record (upsert by model_name + version)."""
    with FileLock(str(REGISTRY_LOCK)):
        records = _read_registry()
        records = [
            r for r in records
            if not (r["model_name"] == record.model_name and r["version"] == record.version)
        ]
        records.append(asdict(record))
        _atomic_write(REGISTRY_PATH, records)


def promote(model_name: str, version: str) -> None:
    """
    Set status=production for `version`; retire previous production version.
    Appends an entry to the promotion log for rollback reference.
    """
    with FileLock(str(REGISTRY_LOCK)):
        records = _read_registry()
        old_prod = next(
            (r for r in records if r["model_name"] == model_name and r["status"] == "production"),
            None,
        )
        new_found = False
        for r in records:
            if r["model_name"] != model_name:
                continue
            if r["version"] == version:
                r["status"] = "production"
                new_found = True
            elif r["status"] == "production":
                r["status"] = "retired"
        if not new_found:
            raise ValueError(
                f"Version '{version}' for model '{model_name}' not found in registry. "
                "Register it before promoting."
            )
        _atomic_write(REGISTRY_PATH, records)

        log: list[dict] = (
            json.loads(PROMOTION_LOG.read_text(encoding="utf-8"))
            if PROMOTION_LOG.exists()
            else []
        )
        log.append({
            "model_name":  model_name,
            "new_version": version,
            "old_version": old_prod["version"] if old_prod else None,
            "promoted_at": datetime.utcnow().isoformat(),
        })
        _atomic_write(PROMOTION_LOG, log)


def rollback(model_name: str) -> bool:
    """
    Promote the most recently retired version back to production.
    Returns True on success, False if no retired version exists.
    """
    records = _read_registry()
    retired = [r for r in records if r["model_name"] == model_name and r["status"] == "retired"]
    if not retired:
        return False
    retired.sort(key=lambda r: r["trained_at"], reverse=True)
    promote(model_name, retired[0]["version"])
    return True


def get_production(model_name: str) -> dict | None:
    """Return the current production record for the model, or None."""
    records = _read_registry()
    return next(
        (r for r in records if r["model_name"] == model_name and r["status"] == "production"),
        None,
    )


def list_versions(model_name: str) -> list[dict]:
    """All records for a model, newest first."""
    records = _read_registry()
    return sorted(
        [r for r in records if r["model_name"] == model_name],
        key=lambda r: r["trained_at"],
        reverse=True,
    )


def get_production_artifact_dir(model_name: str) -> Path | None:
    """
    Resolve the production artifact directory for a model.
    Returns None if no production version is registered yet
    (callers should fall back to the flat artifacts/models/ layout).
    """
    record = get_production(model_name)
    if record is None:
        return None
    return ARTIFACTS_DIR / record["artifact_dir"]
