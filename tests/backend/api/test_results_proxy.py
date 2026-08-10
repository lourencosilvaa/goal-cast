"""``/api/results/*`` — the thin proxy the frontend actually calls.

The browser never talks to the results service directly. Two reasons, and both
are load-bearing: the service key would have to be shipped to the client, and
the user's own authentication would have to be duplicated in a second service.
So the backend keeps its existing user auth, holds the service key server-side
and forwards.

What the proxy must never do is turn a failure into an empty list. A 503 with
a message is a screen that says "results unavailable"; a 200 with no matches
is a screen that says the round was not played.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.results import get_results_gateway, router
from src.backend.core.auth import get_approved_user
from src.backend.services.results_gateway import (
    MissingGatewayConfigError,
    ResultsRequestRejected,
    ResultsServiceUnavailable,
)
from src.contracts.results import (
    HistoryResponse,
    LiveResultsResponse,
    MatchModel,
)

_MATCH = MatchModel(
    league="P1",
    kickoff="2026-08-09T17:00:00",
    home_team="Porto",
    away_team="Alverca",
    status="live",
    home_goals=2,
    away_goals=0,
    minute="67'",
    source="football-data.org",
)


class _Gateway:
    def __init__(self, error=None):
        self._error = error
        self.live_calls: list[list[str]] = []
        self.history_calls: list[tuple[str, str]] = []

    def live(self, leagues):
        self.live_calls.append(list(leagues))
        if self._error:
            raise self._error
        return LiveResultsResponse(
            fetched_at="2026-08-09T17:30:00",
            source="football-data.org",
            stale=False,
            matches=[_MATCH],
            events=[],
        )

    def history(self, league, season):
        self.history_calls.append((league, season))
        if self._error:
            raise self._error
        return HistoryResponse(
            league=league, season=season, source="local-corpus", matches=[_MATCH]
        )


def _client(gateway=None, authenticated: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_results_gateway] = lambda: gateway or _Gateway()
    if authenticated:
        app.dependency_overrides[get_approved_user] = lambda: "test-user"
    return TestClient(app)


class TestAuthentication:
    def test_live_requires_an_approved_user(self):
        assert _client(authenticated=False).get("/api/results/live").status_code == 401

    def test_history_requires_an_approved_user(self):
        response = _client(authenticated=False).get(
            "/api/results/history", params={"league": "P1", "season": "2526"}
        )
        assert response.status_code == 401

    def test_an_unauthenticated_call_never_reaches_the_service(self):
        """The service key must not be spent on a request we already refuse."""
        gateway = _Gateway()
        _client(gateway, authenticated=False).get("/api/results/live")
        assert gateway.live_calls == []


class TestLive:
    def test_a_successful_call_returns_the_service_payload(self):
        payload = _client().get("/api/results/live").json()
        assert payload["matches"][0]["home_team"] == "Porto"

    def test_the_leagues_parameter_is_forwarded(self):
        gateway = _Gateway()
        _client(gateway).get("/api/results/live", params={"leagues": "E0,P1"})
        assert gateway.live_calls == [["E0", "P1"]]

    def test_omitting_the_parameter_forwards_nothing(self):
        gateway = _Gateway()
        _client(gateway).get("/api/results/live")
        assert gateway.live_calls == [[]]

    def test_staleness_reaches_the_frontend(self):
        payload = _client().get("/api/results/live").json()
        assert payload["stale"] is False


class TestHistory:
    def test_a_successful_call_returns_the_service_payload(self):
        response = _client().get(
            "/api/results/history", params={"league": "P1", "season": "2526"}
        )
        assert response.json()["source"] == "local-corpus"

    def test_the_query_is_forwarded(self):
        gateway = _Gateway()
        _client(gateway).get(
            "/api/results/history", params={"league": "E0", "season": "2425"}
        )
        assert gateway.history_calls == [("E0", "2425")]

    def test_the_league_is_required(self):
        response = _client().get("/api/results/history", params={"season": "2526"})
        assert response.status_code == 422


class TestFailures:
    def test_an_unreachable_service_is_a_503(self):
        gateway = _Gateway(ResultsServiceUnavailable("results service unreachable"))
        assert _client(gateway).get("/api/results/live").status_code == 503

    def test_the_503_explains_what_is_unavailable(self):
        gateway = _Gateway(ResultsServiceUnavailable("results service unreachable"))
        detail = _client(gateway).get("/api/results/live").json()["detail"]
        assert "results service" in detail.lower()

    def test_a_failure_is_never_an_empty_list(self):
        gateway = _Gateway(ResultsServiceUnavailable("down"))
        payload = _client(gateway).get("/api/results/live").json()
        assert "matches" not in payload

    def test_history_fails_the_same_way(self):
        gateway = _Gateway(ResultsServiceUnavailable("down"))
        response = _client(gateway).get(
            "/api/results/history", params={"league": "P1", "season": "2526"}
        )
        assert response.status_code == 503

    def test_a_rejected_request_is_a_422_not_a_503(self):
        """The service says the league does not exist; retrying will not help."""
        gateway = _Gateway(ResultsRequestRejected("unknown league 'ZZ'"))
        response = _client(gateway).get("/api/results/live", params={"leagues": "ZZ"})
        assert response.status_code == 422

    def test_a_missing_deployment_variable_is_a_503(self):
        """Our configuration is wrong, which is an outage from where the user
        stands — but the message must name the variable for whoever reads the
        logs."""
        gateway = _Gateway(MissingGatewayConfigError("RESULTS_SERVICE_URL is not set"))
        response = _client(gateway).get("/api/results/live")
        assert response.status_code == 503
        assert "RESULTS_SERVICE_URL" in response.json()["detail"]
