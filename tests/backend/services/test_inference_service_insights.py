"""Tests for the statistics calls InferenceService makes to the HF Space.

These are read-only lookups served from the Space's in-memory history, so they
use the short lookup timeout rather than the long prediction one, and — like
every other call on this service — they refuse to run at all when on-demand
inference is disabled in configuration.
"""

from unittest.mock import MagicMock

import httpx
import pytest

from src.backend.services.inference_service import InferenceService
from tests.backend.services.conftest import FakeAsyncClient, make_response

SPACE_URL = "https://user-space.hf.space"


def _config(enabled: bool = True, space_url: str = SPACE_URL) -> MagicMock:
    config = MagicMock()
    config.inference.enabled = enabled
    config.inference.space_url = space_url
    return config


class TestMatchInsights:

    @pytest.mark.asyncio
    async def test_posts_to_the_match_insights_endpoint(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response({})))
        await InferenceService(_config()).get_match_insights(
            home_team="Sporting", away_team="Porto", league_code="P1"
        )
        assert client.last_url == f"{SPACE_URL}/match-insights"

    @pytest.mark.asyncio
    async def test_sends_the_fixture_in_the_body(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response({})))
        await InferenceService(_config()).get_match_insights(
            home_team="Sporting", away_team="Porto", league_code="P1"
        )
        assert client.last_json == {
            "home_team": "Sporting",
            "away_team": "Porto",
            "league_code": "P1",
        }

    @pytest.mark.asyncio
    async def test_returns_the_payload(self, patch_async_client):
        payload = {"home_team": "Sporting", "head_to_head": {"played": 3}}
        patch_async_client(FakeAsyncClient(make_response(payload)))
        result = await InferenceService(_config()).get_match_insights(
            home_team="Sporting", away_team="Porto", league_code="P1"
        )
        assert result == payload

    @pytest.mark.asyncio
    async def test_uses_the_lookup_timeout(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response({})))
        await InferenceService(_config()).get_match_insights(
            home_team="Sporting", away_team="Porto", league_code="P1"
        )
        assert client.init_kwargs["timeout"] == InferenceService.LOOKUP_TIMEOUT

    @pytest.mark.asyncio
    async def test_upstream_status_error_propagates(self, patch_async_client):
        error = httpx.HTTPStatusError(
            "not found",
            request=httpx.Request("POST", f"{SPACE_URL}/match-insights"),
            response=httpx.Response(404),
        )
        patch_async_client(FakeAsyncClient(make_response({}, status_error=error)))
        with pytest.raises(httpx.HTTPStatusError):
            await InferenceService(_config()).get_match_insights(
                home_team="Sporting", away_team="Nobody", league_code="P1"
            )

    @pytest.mark.asyncio
    async def test_disabled_inference_raises_before_any_request(
        self, patch_async_client
    ):
        client = patch_async_client(FakeAsyncClient(make_response({})))
        with pytest.raises(RuntimeError):
            await InferenceService(_config(enabled=False)).get_match_insights(
                home_team="Sporting", away_team="Porto", league_code="P1"
            )
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_unconfigured_space_url_raises(self, patch_async_client):
        patch_async_client(FakeAsyncClient(make_response({})))
        with pytest.raises(RuntimeError):
            await InferenceService(_config(space_url="")).get_match_insights(
                home_team="Sporting", away_team="Porto", league_code="P1"
            )


class TestTeamInsights:

    @pytest.mark.asyncio
    async def test_gets_the_team_insights_endpoint(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response({})))
        await InferenceService(_config()).get_team_insights(
            team="Sporting", league_code="P1"
        )
        assert client.last_url == f"{SPACE_URL}/team-insights"

    @pytest.mark.asyncio
    async def test_sends_the_query_as_parameters(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response({})))
        await InferenceService(_config()).get_team_insights(
            team="Sporting", league_code="P1"
        )
        assert client.last_call[2]["params"] == {
            "team": "Sporting",
            "league_code": "P1",
        }

    @pytest.mark.asyncio
    async def test_returns_the_payload(self, patch_async_client):
        payload = {"team": "Sporting", "overall": {"played": 5}}
        patch_async_client(FakeAsyncClient(make_response(payload)))
        result = await InferenceService(_config()).get_team_insights(
            team="Sporting", league_code="P1"
        )
        assert result == payload

    @pytest.mark.asyncio
    async def test_uses_the_lookup_timeout(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response({})))
        await InferenceService(_config()).get_team_insights(
            team="Sporting", league_code="P1"
        )
        assert client.init_kwargs["timeout"] == InferenceService.LOOKUP_TIMEOUT

    @pytest.mark.asyncio
    async def test_trailing_slash_in_space_url_is_normalised(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response({})))
        await InferenceService(_config(space_url=f"{SPACE_URL}/")).get_team_insights(
            team="Sporting", league_code="P1"
        )
        assert client.last_url == f"{SPACE_URL}/team-insights"

    @pytest.mark.asyncio
    async def test_transport_error_propagates(self, patch_async_client):
        patch_async_client(
            FakeAsyncClient(request_error=httpx.ConnectError("space down"))
        )
        with pytest.raises(httpx.ConnectError):
            await InferenceService(_config()).get_team_insights(
                team="Sporting", league_code="P1"
            )

    @pytest.mark.asyncio
    async def test_disabled_inference_raises_before_any_request(
        self, patch_async_client
    ):
        client = patch_async_client(FakeAsyncClient(make_response({})))
        with pytest.raises(RuntimeError):
            await InferenceService(_config(enabled=False)).get_team_insights(
                team="Sporting", league_code="P1"
            )
        assert client.calls == []
