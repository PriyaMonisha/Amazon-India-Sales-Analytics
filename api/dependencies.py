"""
Shared FastAPI dependencies — auth and rate limiting.

Centralised here to avoid circular imports: api/main.py imports routers,
so routers must NOT import from api/main.py. Import from this module instead.
"""
from __future__ import annotations

import os

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_API_KEY_HEADER)) -> str:
    """Validate X-API-Key header against API_KEY env var. Raises 403 on mismatch."""
    expected = os.getenv("API_KEY", "")
    if not expected or api_key != expected:
        raise HTTPException(status_code=403, detail="Forbidden")
    return api_key


limiter = Limiter(key_func=get_remote_address)
