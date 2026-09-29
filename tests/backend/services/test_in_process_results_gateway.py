"""The in-process gateway: the results service without the HTTP hop.

It must speak the same contract and the same failure vocabulary as the HTTP
gateway, so the routers above cannot tell which one they were given.
"""

from datetime import datetime

import pytest

from config.config_loader import ResultsLiveConfig
from src.backend.services.results_gateway import (
    InProcessResultsGateway,
    ResultsRequestRejected,
    ResultsServiceUnavailable,
)
from src.results_service.service import ResultsCatalogue, ResultsService
from src.scrapers.results.live_tracker import LiveResultsTracker
from src.scrapers.results.models import MatchResult, MatchStatus

_CATALOGUE = ResultsCatalogue(leagues=("P1", "E0"), seasons=("2526",))


def _match() -> MatchResult:
    return MatchResult(
        league="P1",
        kickoff=datetime(2026, 8, 9, 17, 0),
        home_team="Porto",
        away_team="Alverca",
        status=MatchStatus.FINISHED,
        home_goals=1,
        away_goals=0,
        source="stub",
    )


class _History:
    last_source = "local-corpus"

    def fetch_history(self, query):
        return [_match()]


class _Live:
    last_source = "football-data.org"

    def __init__(self):
        self.calls = 0

    def fetch_live(self, leagues):
        self.calls += 1
        return [_match()]


def _service(live=None) -> ResultsService:
    tracker = LiveResultsTracker(
        live or _Live(),
        ResultsLiveConfig(poll_interval_seconds=60, stale_after_seconds=300),
        clock=lambda: datetime(2026, 8, 9, 17, 30),
    )
    return ResultsService(_History(), tracker, _CATALOGUE)


class TestInProcessGateway:
    def test_live_returns_the_shared_contract(self):
        gateway = InProcessResultsGateway(_service)
        response = gateway.live(["P1"])
        assert response.source == "football-data.org"
        assert response.matches[0].home_team == "Porto"

    def test_history_returns_the_shared_contract(self):
        gateway = InProcessResultsGateway(_service)
        response = gateway.history("P1", "2526")
        assert (response.league, response.season) == ("P1", "2526")
        assert response.source == "local-corpus"

    def test_unknown_league_is_a_rejection_not_an_empty_list(self):
        gateway = InProcessResultsGateway(_service)
        with pytest.raises(ResultsRequestRejected, match="unknown league"):
            gateway.live(["ZZ"])

    def test_unknown_season_is_a_rejection(self):
        gateway = InProcessResultsGateway(_service)
        with pytest.raises(ResultsRequestRejected, match="unknown season"):
            gateway.history("P1", "0001")

    def test_provider_failure_is_unavailability(self):
        class _Broken:
            def live(self, leagues):
                raise RuntimeError("boom")

        gateway = InProcessResultsGateway(_Broken)
        with pytest.raises(ResultsServiceUnavailable, match="boom"):
            gateway.live(["P1"])

    def test_the_service_is_built_once_so_the_ttl_cache_survives(self):
        built: list[int] = []
        live = _Live()
        service = _service(live)

        def factory():
            built.append(1)
            return service

        gateway = InProcessResultsGateway(factory)
        gateway.live(["P1"])
        gateway.live(["P1"])
        assert len(built) == 1
        assert live.calls == 1  # the second call was served from the TTL cache

    def test_a_failed_build_is_stated_and_retried(self):
        attempts: list[int] = []

        def factory():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("FOOTBALL_DATA_API_KEY missing")
            return _service()

        gateway = InProcessResultsGateway(factory)
        with pytest.raises(ResultsServiceUnavailable, match="FOOTBALL_DATA"):
            gateway.live(["P1"])
        assert gateway.live(["P1"]).matches  # recovers without a restart
