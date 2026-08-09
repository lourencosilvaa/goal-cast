"""The one place this project makes an outbound HTTP call for fixtures.

Deliberately our own thin wrapper over ``requests`` rather than any vendor SDK:
the call shapes below are copied from each provider's official samples, so what
goes on the wire is visible in this repository and does not change because a
third-party package was upgraded.

Centralising it also means timeout and retry policy is stated once, in
configuration, instead of being re-decided in each provider.
"""

import time
from typing import Any, ClassVar

import requests


class HttpTransport:
    """Performs GETs with a timeout and bounded retries.

    Retries only on transport-level errors, never on an HTTP status. A 403 from
    football-data.org means "your plan does not cover this competition" and a
    401 from The Odds API means "bad key" — repeating either wastes quota and
    changes nothing. Only a connection that never produced a response is worth
    trying again, which is not hypothetical: api.football-data.org returned a
    spurious SSL EOF once during development and succeeded on retry.
    """

    #: Errors worth retrying — the request never reached a server.
    RETRYABLE: ClassVar[tuple[type[Exception], ...]] = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    )

    def __init__(
        self, timeout: int = 15, retries: int = 2, backoff_seconds: float = 1.0
    ) -> None:
        self._timeout = timeout
        self._retries = retries
        self._backoff = backoff_seconds

    def get(
        self, url: str, params: dict | None = None, headers: dict | None = None
    ) -> Any:
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                return requests.get(
                    url,
                    params=params or {},
                    headers=headers or {},
                    timeout=self._timeout,
                )
            except self.RETRYABLE as exc:
                last = exc
                if attempt < self._retries:
                    time.sleep(self._backoff * (attempt + 1))
        assert last is not None
        raise last
