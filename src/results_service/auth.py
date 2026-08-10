"""Service-to-service authentication.

The same shape as ``RETRAIN_API_KEY`` elsewhere in this project — a shared
secret in a header — with one deliberate difference: **an unconfigured key
stops the service from starting** rather than making the endpoint answer 503.

That difference is the whole point. This service exists to run a scraper and
spend a metered API quota; a deployment that forgot the variable would
otherwise sit on the public internet doing both for anyone who found it. A
process that refuses to boot is noticed in minutes. An open one is not noticed
at all (§7.4 — no silent fallbacks).
"""

import hmac
import os
from typing import Any, ClassVar, Mapping

from fastapi import Header, HTTPException, Request, status


class MissingServiceKeyError(RuntimeError):
    """No service key was configured, so the service must not start."""


def service_api_key(config: Any, env: Mapping[str, str] | None = None) -> str:
    """The shared secret, read from the environment variable config names.

    ``env`` is injected so a test states the environment it runs in rather
    than inheriting the developer's.
    """
    source = os.environ if env is None else env
    name = config.api_key_env
    key = str(source.get(name, "") or "").strip()
    if not key:
        raise MissingServiceKeyError(
            f"{name} is not set. The results service will not start without a "
            f"service key: it would otherwise expose a scraper and a metered "
            f"API quota to anyone who found its URL."
        )
    return key


class ApiKeyGuard:
    """Compares a request's key against the configured one."""

    HEADER: ClassVar[str] = "X-API-Key"

    def __init__(self, expected: str) -> None:
        if not expected:
            raise MissingServiceKeyError(
                "an ApiKeyGuard with no key would accept unauthenticated calls"
            )
        self._expected = expected

    def verify(self, provided: str | None) -> None:
        """Raise 401 unless ``provided`` is the configured key.

        Constant-time comparison: a plain ``==`` returns as soon as two bytes
        differ, which leaks the length of the shared prefix to anyone timing
        the response.
        """
        if not provided or not hmac.compare_digest(provided, self._expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                # Says nothing about what was expected, or about whether the
                # header was absent versus wrong.
                detail="Invalid or missing service API key.",
            )


def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None)
) -> None:
    """FastAPI dependency: the guard lives on the app, not in a global.

    Reading it from ``app.state`` is what lets the app factory own the key and
    lets a test build an app with a different one, without a module-level
    singleton that the import order could decide.
    """
    guard: ApiKeyGuard = request.app.state.api_key_guard
    guard.verify(x_api_key)
