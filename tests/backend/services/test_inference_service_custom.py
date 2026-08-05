"""Tests for InferenceService.predict_custom() and get_teams()."""

from unittest.mock import MagicMock

import httpx
import pytest

from src.backend.services.inference_service import InferenceService
from tests.backend.services.conftest import FakeAsyncClient, make_response

_CUSTOM_RESULT = {
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "predicted_outcome": "Home Win",
    "confidence": 0.65,
    "probabilities": {"home_win": 0.65, "draw": 0.2, "away_win": 0.15},
    "league": "Premier League",
}


def _make_config(
    enabled: bool = True, space_url: str = "https://user.hf.space"
) -> MagicMock:
    config = MagicMock()
    config.inference.enabled = enabled
    config.inference.space_url = space_url
    return config


class TestInferenceServicePredictCustom:

    @pytest.mark.asyncio
    async def test_calls_hf_space_predict_custom(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response(_CUSTOM_RESULT)))
        svc = InferenceService(_make_config())
        result = await svc.predict_custom("Arsenal", "Chelsea", "E0")

        assert client.last_url == "https://user.hf.space/predict-custom"
        assert result["predicted_outcome"] == "Home Win"

    @pytest.mark.asyncio
    async def test_sends_correct_payload(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response(_CUSTOM_RESULT)))
        svc = InferenceService(_make_config())
        await svc.predict_custom("Arsenal", "Chelsea", "E0")

        assert client.last_json == {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "league_code": "E0",
        }

    @pytest.mark.asyncio
    async def test_disabled_raises_runtime_error(self):
        svc = InferenceService(_make_config(enabled=False))
        with pytest.raises(RuntimeError, match="disabled"):
            await svc.predict_custom("Arsenal", "Chelsea", "E0")

    @pytest.mark.asyncio
    async def test_hf_space_error_propagates(self, patch_async_client):
        patch_async_client(
            FakeAsyncClient(
                make_response(
                    status_error=httpx.HTTPStatusError(
                        "500",
                        request=httpx.Request("POST", "https://user.hf.space"),
                        response=httpx.Response(500),
                    )
                )
            )
        )
        svc = InferenceService(_make_config())
        with pytest.raises(httpx.HTTPStatusError):
            await svc.predict_custom("Arsenal", "Chelsea", "E0")


class TestInferenceServiceGetTeams:
    """get_teams() backs the RemoteTeamRepository behind /api/predictions/teams."""

    @pytest.mark.asyncio
    async def test_gets_teams_endpoint(self, patch_async_client):
        client = patch_async_client(
            FakeAsyncClient(make_response({"E0": ["Arsenal", "Chelsea"]}))
        )
        svc = InferenceService(_make_config())
        result = await svc.get_teams()

        assert client.last_call[0] == "GET"
        assert client.last_url == "https://user.hf.space/teams"
        assert result == {"E0": ["Arsenal", "Chelsea"]}

    @pytest.mark.asyncio
    async def test_disabled_raises_runtime_error(self):
        svc = InferenceService(_make_config(enabled=False))
        with pytest.raises(RuntimeError, match="disabled"):
            await svc.get_teams()

    @pytest.mark.asyncio
    async def test_empty_space_response_is_returned_verbatim(self, patch_async_client):
        """Interpreting {} is the repository's job, not the service's."""
        patch_async_client(FakeAsyncClient(make_response({})))
        svc = InferenceService(_make_config())
        assert await svc.get_teams() == {}

    @pytest.mark.asyncio
    async def test_connection_error_propagates(self, patch_async_client):
        patch_async_client(
            FakeAsyncClient(request_error=httpx.ConnectError("unreachable"))
        )
        svc = InferenceService(_make_config())
        with pytest.raises(httpx.ConnectError):
            await svc.get_teams()
