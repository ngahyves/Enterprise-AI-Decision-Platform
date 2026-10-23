# app/auth.py
"""API key authentication for protected routes."""

import os
from fastapi import Header, HTTPException, status

# Read once at import time
_API_KEY = os.getenv("API_KEY", "")


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """FastAPI dependency — attach to any route that requires auth."""
    if not _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured on server",
        )
    if x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key
