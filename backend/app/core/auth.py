"""
Optional API-key authentication.

When settings.API_KEY is non-empty, every request must carry a matching
X-API-Key header, otherwise a 401 is returned. When API_KEY is empty
(the local-dev default), all requests are allowed.
"""

import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config.settings import settings

_api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """FastAPI dependency enforcing the optional X-API-Key header."""
    if not settings.API_KEY:
        # Auth disabled (local dev default)
        return

    if not api_key or not secrets.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(status_code=401, detail='Invalid or missing API key')
