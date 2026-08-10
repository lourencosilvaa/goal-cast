"""``GET /api/in-play`` — the live re-pricing the dashboard rows expand into.

The endpoint owns three things worth testing at this level: it is behind the
same authentication as everything else, it answers in the shape the frontend
joins on, and it fails the way the rest of the results stack fails — a 503
that says so, never a 200 with an empty board. The pricing itself is tested in
``tests/backend/services/test_in_play.py`` and the joining in
``test_in_play_board.py``; neither is re-tested here.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.in_play import get_in_play_board, router
from src.backend.core.auth import get_approved_user
from src.backend.services.in_play import ExpectedGoals, InPlayCalculator, LiveState
from src.backend.services.in_play_board import (
    InPlayBoard,
    InPlayBoardResult,
    PricedMatch,
    UnpricedMatch,
)
from src.backend.services.results_gateway import (
    MissingGatewayConfigError,
    ResultsRequestRejected,
    ResultsServiceUnavailable,
)
from src.contracts.results import LiveResultsResponse, MatchModel
from src.models.outcome_model import OutcomeProbabilities

_FORECAST = InPlayCalculator().forecast(
    ExpectedGoals(home=2.1, away=0.8),
    LiveState(home_goals=1, away_goals=0, elapsed_minutes=60),
)

_PRICED = PricedMatch(
    league="P1",
    home_team="Porto",
    away_team="Alverca",
    home_goals=1,
    away_goals=0,
    elapsed_minutes=60,
    minute_estimated=True,
    status="live",
    pre_match=OutcomeProbabilities(home_win=0.72, draw=0.18, away_win=0.10),
    forecast=_FORECAST,
)

_UNPRICED = UnpricedMatch(
    league="P1", home_team="Sporting CP", away_team="Braga", reason=InPlayBoard.UNKNOWN_TEAM
)


class _Board:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls: list[list[str]] = []

    def build(self, leagues):
        self.calls.append(list(leagues))
        if self._error:
            raise self._error
        return self._result or InPlayBoardResult(
            fetched_at="2026-08-09T17:30:00",
            stale=False,
            matches=[_PRICED],
            unpriced=[_UNPRICED],
        )


class _LiveGateway:
    """One live fixture, spelled the way the live provider spells it."""

    def live(self, leagues):
        return LiveResultsResponse(
            fetched_at="2026-08-09T17:30:00",
            matches=[
                MatchModel(
                    league="P1",
                    kickoff="2026-08-09T17:00:00",
                    home_team="Sporting CP",
                    away_team="Porto",
                    status="live",
                    home_goals=1,
                    away_goals=0,
                    elapsed_minutes=60,
                )
            ],
        )


class _PredictionService:
    """Stands in for Supabase with one stored row for that fixture."""

    def get_league_predictions(self, league_code, target_date=None):
        class _Rows:
            matches = [
                {
                    "home_team": "Sp Lisbon",
                    "away_team": "Porto",
                    "probabilities": {"home_win": 0.5, "draw": 0.25, "away_win": 0.25},
                    "expected_goals": {"home": 1.6, "away": 1.2, "total": 2.8},
                }
            ]

        return _Rows()


def _client(board=None, authenticated: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_in_play_board] = lambda: board or _Board()
    if authenticated:
        app.dependency_overrides[get_approved_user] = lambda: "test-user"
    return TestClient(app)


class TestAuthentication:
    def test_it_requires_an_approved_user(self):
        assert _client(authenticated=False).get("/api/in-play").status_code == 401

    def test_an_unauthenticated_call_never_reaches_the_results_service(self):
        board = _Board()
        _client(board, authenticated=False).get("/api/in-play")
        assert board.calls == []


class TestThePayload:
    def test_a_priced_match_is_returned(self):
        payload = _client().get("/api/in-play").json()
        assert payload["matches"][0]["home_team"] == "Porto"

    def test_the_live_probabilities_are_present(self):
        match = _client().get("/api/in-play").json()["matches"][0]
        assert match["live"]["home_win"] > match["pre_match"]["home_win"]

    def test_the_pre_match_probabilities_are_carried_alongside(self):
        """Shown together: the movement between them is what the card says."""
        match = _client().get("/api/in-play").json()["matches"][0]
        assert match["pre_match"]["home_win"] == 0.72

    def test_the_remaining_time_is_reported_in_minutes(self):
        """A fraction is the model's unit; a reader's unit is minutes."""
        assert _client().get("/api/in-play").json()["matches"][0]["remaining_minutes"] == 30

    def test_the_expected_final_score_is_reported(self):
        match = _client().get("/api/in-play").json()["matches"][0]
        assert match["expected_home_goals"] > 1.0

    def test_the_scoreboard_is_reported(self):
        match = _client().get("/api/in-play").json()["matches"][0]
        assert (match["home_goals"], match["away_goals"]) == (1, 0)

    def test_an_estimated_minute_is_flagged(self):
        assert _client().get("/api/in-play").json()["matches"][0]["minute_estimated"] is True

    def test_matches_that_could_not_be_priced_are_listed_with_a_reason(self):
        """A shorter list than the number of matches being played has to be
        explainable — usually a team name nobody has approved yet."""
        unpriced = _client().get("/api/in-play").json()["unpriced"][0]
        assert unpriced["reason"] == "unknown_team"
        assert unpriced["home_team"] == "Sporting CP"

    def test_staleness_reaches_the_frontend(self):
        assert _client().get("/api/in-play").json()["stale"] is False

    def test_nothing_in_progress_is_an_empty_board_not_an_error(self):
        board = _Board(InPlayBoardResult(fetched_at="2026-08-09T17:30:00"))
        response = _client(board).get("/api/in-play")
        assert response.status_code == 200 and response.json()["matches"] == []


class TestTheLeagueFilter:
    def test_the_leagues_parameter_is_forwarded(self):
        board = _Board()
        _client(board).get("/api/in-play", params={"leagues": "P1,E0"})
        assert board.calls == [["P1", "E0"]]

    def test_omitting_it_asks_for_everything(self):
        board = _Board()
        _client(board).get("/api/in-play")
        assert board.calls == [[]]


class TestTheWiring:
    """The dependency the route uses in production, exercised once.

    Every other test overrides it, so without this the only place the board is
    ever assembled from the shipped registry and alias seed is a running
    container — and a wrong path there prices nothing, silently.
    """

    @staticmethod
    def _request():
        from config.config_loader import load_config

        app = FastAPI()
        app.state.config = load_config()
        app.state.prediction_service = _PredictionService()

        class _Request:
            def __init__(self, application):
                self.app = application

        return _Request(app)

    @staticmethod
    def _without_supabase(monkeypatch, error: Exception | None = None):
        """Keep the real client out of it.

        ``get_supabase_client`` is ``lru_cache``d process-wide, so building one
        here would leak a live client into every later test that asserts what
        happens when Supabase is absent.
        """
        import src.backend.api.in_play as module

        def client():
            if error:
                raise error
            return object()

        monkeypatch.setattr(module, "get_supabase_client", client)

    def test_it_prices_a_match_using_the_shipped_registry_and_seed(self, monkeypatch):
        """"Sporting CP" is how the live provider spells Sp Lisbon; resolving
        it needs the registry file *and* the alias seed to be found."""
        self._without_supabase(monkeypatch)
        board = get_in_play_board(self._request(), gateway=_LiveGateway())
        assert board.build([]).matches[0].home_team == "Sp Lisbon"

    def test_it_survives_an_unreachable_alias_store(self, monkeypatch):
        """Losing admin-approved aliases costs a few names; it must not cost
        the endpoint. Only the committed seed remains, which is enough here."""
        self._without_supabase(monkeypatch, RuntimeError("SUPABASE_URL is not set"))
        board = get_in_play_board(self._request(), gateway=_LiveGateway())
        assert board.build([]).matches[0].home_team == "Sp Lisbon"


class TestFailures:
    def test_an_unreachable_results_service_is_a_503(self):
        board = _Board(error=ResultsServiceUnavailable("results service unreachable"))
        assert _client(board).get("/api/in-play").status_code == 503

    def test_a_failure_is_never_an_empty_board(self):
        """"No live matches" and "we could not find out" are opposite screens."""
        board = _Board(error=ResultsServiceUnavailable("down"))
        assert "matches" not in _client(board).get("/api/in-play").json()

    def test_a_missing_deployment_variable_names_itself(self):
        board = _Board(error=MissingGatewayConfigError("RESULTS_SERVICE_URL is not set"))
        response = _client(board).get("/api/in-play")
        assert response.status_code == 503
        assert "RESULTS_SERVICE_URL" in response.json()["detail"]

    def test_a_rejected_league_is_a_422(self):
        board = _Board(error=ResultsRequestRejected("unknown league 'ZZ'"))
        response = _client(board).get("/api/in-play", params={"leagues": "ZZ"})
        assert response.status_code == 422
