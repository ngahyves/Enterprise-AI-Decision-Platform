# app/limiter.py
"""Rate limiter shared between main and routes.

Key function prefers the X-API-Key header over IP:
- Two clients sharing an IP (NAT) keep separate quotas.
- A compromised key is throttled independently of IP rotation.
"""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def rate_limit_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    return api_key if api_key else get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key, default_limits=["10/minute"])
