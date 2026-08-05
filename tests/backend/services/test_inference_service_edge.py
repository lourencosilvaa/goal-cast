"""Edge case tests for the InferenceService async httpx client."""

from unittest.mock import MagicMock

import httpx
import pytest

from src.backend.services.inference_service import InferenceService
from tests.backend.services.conftest import FakeAsyncClient, make_response

_EXPECTED_TIMEOUT_SECONDS = 120


def _make_config(space_url: str = "https://user-space.hf.space") -> MagicMock:
    config = MagicMock()
    config.inference.enabled = True
    config.inference.space_url = space_url
    return config


class TestInferenceServiceEdgeCases:

    @pytest.mark.asyncio
    async def test_trailing_slash_in_space_url_is_stripped(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response({"predictions": []})))
        svc = InferenceService(_make_config("https://user-space.hf.space/"))
        await svc.run(target_date="28/04/2024")

        assert client.last_url == "https://user-space.hf.space/infer"

    @pytest.mark.asyncio
    async def test_space_connection_error_propagates(self, patch_async_client):
        patch_async_client(
            FakeAsyncClient(request_error=httpx.ConnectError("unreachable"))
        )
        svc = InferenceService(_make_config())

        with pytest.raises(httpx.ConnectError):
            await svc.run(target_date="28/04/2024")

    @pytest.mark.asyncio
    async def test_space_timeout_propagates(self, patch_async_client):
        patch_async_client(
            FakeAsyncClient(request_error=httpx.ReadTimeout("timed out"))
        )
        svc = InferenceService(_make_config())

        with pytest.raises(httpx.ReadTimeout):
            await svc.run(target_date="28/04/2024")

    @pytest.mark.asyncio
    async def test_space_500_raises_http_error(self, patch_async_client):
        patch_async_client(
            FakeAsyncClient(
                make_response(
                    status_error=httpx.HTTPStatusError(
                        "500 Server Error",
                        request=httpx.Request("POST", "https://user-space.hf.space"),
                        response=httpx.Response(500),
                    )
                )
            )
        )
        svc = InferenceService(_make_config())

        with pytest.raises(httpx.HTTPStatusError):
            await svc.run(target_date="28/04/2024")

    @pytest.mark.asyncio
    async def test_missing_predictions_key_returns_empty_list(
        self, patch_async_client
    ):
        patch_async_client(FakeAsyncClient(make_response({})))
        svc = InferenceService(_make_config())
        assert await svc.run(target_date="28/04/2024") == []

    @pytest.mark.asyncio
    async def test_request_uses_120s_timeout(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response({"predictions": []})))
        svc = InferenceService(_make_config())
        await svc.run()

        assert client.init_kwargs.get("timeout") == _EXPECTED_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_disabled_with_space_url_still_raises(self):
        config = _make_config("https://user-space.hf.space")
        config.inference.enabled = False
        svc = InferenceService(config)

        with pytest.raises(RuntimeError, match="disabled"):
            await svc.run()
