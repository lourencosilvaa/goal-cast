"""Shared doubles for the HTTP-backed backend services.

``InferenceService`` talks to the HuggingFace Space through
``httpx.AsyncClient`` used as an async context manager::

    async with httpx.AsyncClient(timeout=...) as client:
        response = await client.post(url, json=...)

:class:`FakeAsyncClient` is a drop-in for that client. It records the
constructor kwargs and every request, so tests can assert on the URL, payload
and timeout without any network access.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest


class FakeAsyncClient:
    """Stands in for ``httpx.AsyncClient`` as an async context manager."""

    def __init__(
        self,
        response: Any = None,
        request_error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.request_error = request_error
        self.init_kwargs: dict[str, Any] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    # -- request recording -------------------------------------------------

    async def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        self.calls.append((method, url, kwargs))
        if self.request_error is not None:
            raise self.request_error
        return self.response

    async def post(self, url: str, **kwargs: Any) -> Any:
        return await self._request("POST", url, **kwargs)

    async def get(self, url: str, **kwargs: Any) -> Any:
        return await self._request("GET", url, **kwargs)

    # -- assertions helpers ------------------------------------------------

    @property
    def last_call(self) -> tuple[str, str, dict[str, Any]]:
        assert self.calls, "no request was made"
        return self.calls[-1]

    @property
    def last_url(self) -> str:
        return self.last_call[1]

    @property
    def last_json(self) -> dict[str, Any]:
        return self.last_call[2]["json"]

    # -- async context manager --------------------------------------------

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


def make_response(
    payload: Any = None,
    status_error: BaseException | None = None,
    status_code: int = 200,
) -> MagicMock:
    """Build an httpx-like response returning ``payload`` from ``.json()``.

    ``status_code`` is stated rather than left as a MagicMock attribute: a
    service that inspects the code before calling ``raise_for_status`` — as
    ``predict_custom`` does, to tell a refusal from an outage — would otherwise
    compare against something truthy and meaningless.
    """
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    if status_error is None:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = status_error
    return response


@pytest.fixture
def patch_async_client(monkeypatch):
    """Install a :class:`FakeAsyncClient` in place of ``httpx.AsyncClient``."""

    def _patch(client: FakeAsyncClient) -> FakeAsyncClient:
        def _factory(**kwargs: Any) -> FakeAsyncClient:
            client.init_kwargs = kwargs
            return client

        monkeypatch.setattr(
            "src.backend.services.inference_service.httpx.AsyncClient", _factory
        )
        return client

    return _patch
