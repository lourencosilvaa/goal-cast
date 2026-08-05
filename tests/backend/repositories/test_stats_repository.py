"""Tests for the statistics repository backing /api/stats.

The backend image carries no pandas, so the repository is a pure adapter over
the HuggingFace Space. Its whole job is translating transport failures into
domain errors the API layer can map onto status codes:

* upstream 404          → :class:`TeamNotFoundError`  (→ HTTP 404)
* anything else / down  → :class:`StatsUnavailableError` (→ HTTP 503)
"""

from unittest.mock import AsyncMock

import httpx
import pytest

from src.backend.repositories.stats_repository import (
    MatchStatsQuery,
    RemoteStatsRepository,
    StatsRepository,
    StatsUnavailableError,
    TeamNotFoundError,
    TeamStatsQuery,
)

TEAM_QUERY = TeamStatsQuery(league_code="P1", team="Sporting")
MATCH_QUERY = MatchStatsQuery(league_code="P1", home_team="Sporting", away_team="Porto")


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://space.test/team-insights")
    return httpx.HTTPStatusError(
        "upstream error", request=request, response=httpx.Response(status_code)
    )


class TestInterface:

    def test_remote_repository_implements_the_abstraction(self):
        assert issubclass(RemoteStatsRepository, StatsRepository)

    def test_abstraction_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            StatsRepository()  # type: ignore[abstract]


class TestTeamStats:

    @pytest.mark.asyncio
    async def test_returns_upstream_payload(self):
        service = AsyncMock()
        service.get_team_insights.return_value = {"team": "Sporting"}
        repo = RemoteStatsRepository(service)
        assert await repo.get_team_stats(TEAM_QUERY) == {"team": "Sporting"}

    @pytest.mark.asyncio
    async def test_passes_query_fields_to_the_service(self):
        service = AsyncMock()
        service.get_team_insights.return_value = {}
        await RemoteStatsRepository(service).get_team_stats(TEAM_QUERY)
        service.get_team_insights.assert_awaited_once_with(
            team="Sporting", league_code="P1"
        )

    @pytest.mark.asyncio
    async def test_upstream_404_raises_team_not_found(self):
        service = AsyncMock()
        service.get_team_insights.side_effect = _http_error(404)
        with pytest.raises(TeamNotFoundError):
            await RemoteStatsRepository(service).get_team_stats(TEAM_QUERY)

    @pytest.mark.asyncio
    async def test_upstream_500_raises_stats_unavailable(self):
        service = AsyncMock()
        service.get_team_insights.side_effect = _http_error(500)
        with pytest.raises(StatsUnavailableError):
            await RemoteStatsRepository(service).get_team_stats(TEAM_QUERY)

    @pytest.mark.asyncio
    async def test_transport_failure_raises_stats_unavailable(self):
        service = AsyncMock()
        service.get_team_insights.side_effect = httpx.ConnectError("space down")
        with pytest.raises(StatsUnavailableError):
            await RemoteStatsRepository(service).get_team_stats(TEAM_QUERY)

    @pytest.mark.asyncio
    async def test_disabled_inference_raises_stats_unavailable(self):
        service = AsyncMock()
        service.get_team_insights.side_effect = RuntimeError("inference disabled")
        with pytest.raises(StatsUnavailableError):
            await RemoteStatsRepository(service).get_team_stats(TEAM_QUERY)


class TestMatchStats:

    @pytest.mark.asyncio
    async def test_returns_upstream_payload(self):
        service = AsyncMock()
        service.get_match_insights.return_value = {"home_team": "Sporting"}
        repo = RemoteStatsRepository(service)
        assert await repo.get_match_stats(MATCH_QUERY) == {"home_team": "Sporting"}

    @pytest.mark.asyncio
    async def test_passes_query_fields_to_the_service(self):
        service = AsyncMock()
        service.get_match_insights.return_value = {}
        await RemoteStatsRepository(service).get_match_stats(MATCH_QUERY)
        service.get_match_insights.assert_awaited_once_with(
            home_team="Sporting", away_team="Porto", league_code="P1"
        )

    @pytest.mark.asyncio
    async def test_upstream_404_raises_team_not_found(self):
        service = AsyncMock()
        service.get_match_insights.side_effect = _http_error(404)
        with pytest.raises(TeamNotFoundError):
            await RemoteStatsRepository(service).get_match_stats(MATCH_QUERY)

    @pytest.mark.asyncio
    async def test_upstream_503_raises_stats_unavailable(self):
        service = AsyncMock()
        service.get_match_insights.side_effect = _http_error(503)
        with pytest.raises(StatsUnavailableError):
            await RemoteStatsRepository(service).get_match_stats(MATCH_QUERY)

    @pytest.mark.asyncio
    async def test_transport_failure_raises_stats_unavailable(self):
        service = AsyncMock()
        service.get_match_insights.side_effect = httpx.ReadTimeout("slow space")
        with pytest.raises(StatsUnavailableError):
            await RemoteStatsRepository(service).get_match_stats(MATCH_QUERY)
