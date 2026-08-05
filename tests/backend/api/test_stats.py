"""Tests for the /api/stats endpoints.

The router owns no statistics logic — it validates the upstream payload against
an explicit response contract and maps domain errors onto status codes. Route
placement (never shadowed by ``GET /api/predictions/{league_code}``) is guarded
in ``test_route_resolution.py``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api import stats as stats_module
from src.backend.api.stats import get_stats_repository, router
from src.backend.core.auth import get_current_user
from src.backend.repositories.stats_repository import (
    RemoteStatsRepository,
    StatsUnavailableError,
    TeamNotFoundError,
)


def team_record(**overrides) -> dict:
    record = {
        "played": 5,
        "wins": 2,
        "draws": 2,
        "losses": 1,
        "goals_for": 6,
        "goals_against": 5,
        "points": 8,
        "points_per_game": 1.6,
        "win_pct": 0.4,
        "draw_pct": 0.4,
        "loss_pct": 0.2,
        "avg_goals_for": 1.2,
        "avg_goals_against": 1.0,
        "goal_difference": 1,
    }
    record.update(overrides)
    return record


def match_record(**overrides) -> dict:
    record = {
        "date": "2024-01-29",
        "home_team": "Sporting",
        "away_team": "Porto",
        "home_goals": 1,
        "away_goals": 1,
        "result": "D",
        "venue": "H",
    }
    record.update(overrides)
    return record


def team_payload(team: str = "Sporting") -> dict:
    return {
        "team": team,
        "league": "Liga Portugal",
        "league_code": "P1",
        "overall": team_record(),
        "home": team_record(played=3, wins=2, draws=1, losses=0),
        "away": team_record(played=2, wins=0, draws=1, losses=1),
        "recent": team_record(),
        "form_sequence": ["D", "L", "W", "D", "W"],
        "rates": {
            "clean_sheets": 0.2,
            "failed_to_score": 0.4,
            "btts": 0.6,
            "over_2_5": 0.4,
        },
        "averages": {
            "shots": 10.0,
            "shots_on_target": 4.0,
            "corners": 5.0,
            "cards": 2.0,
        },
        "recent_matches": [match_record()],
    }


def match_payload(goal_markets: dict | None = None) -> dict:
    return {
        "home_team": "Sporting",
        "away_team": "Porto",
        "league": "Liga Portugal",
        "league_code": "P1",
        "head_to_head": {
            "home_team": "Sporting",
            "away_team": "Porto",
            "played": 3,
            "home_wins": 1,
            "draws": 1,
            "away_wins": 1,
            "home_goals": 3,
            "away_goals": 4,
            "avg_goals_home": 1.0,
            "avg_goals_away": 1.33,
            "avg_goals_total": 2.33,
            "btts_pct": 0.67,
            "over_2_5_pct": 0.33,
            "matches": [match_record()],
        },
        "home": team_payload("Sporting"),
        "away": team_payload("Porto"),
        "goal_markets": goal_markets,
    }


GOAL_MARKETS = {
    "expected_goals": {"home": 1.8, "away": 1.1, "total": 2.9},
    "over_under": {
        "over_1_5": 0.78,
        "over_2_5": 0.55,
        "over_3_5": 0.31,
        "under_2_5": 0.45,
    },
    "btts": {"yes": 0.52, "no": 0.48},
    "top_scorelines": [{"score": "2-1", "prob": 0.11}],
    "source": "model",
}


def client(repository) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: "user-123"
    app.dependency_overrides[get_stats_repository] = lambda: repository
    return TestClient(app, raise_server_exceptions=False)


class TestTeamStatsEndpoint:

    def test_returns_team_payload(self):
        repo = AsyncMock()
        repo.get_team_stats.return_value = team_payload()
        res = client(repo).get("/api/stats/team", params={"league_code": "P1", "team": "Sporting"})
        assert res.status_code == 200
        body = res.json()
        assert body["team"] == "Sporting"
        assert body["overall"]["wins"] == 2
        assert body["form_sequence"] == ["D", "L", "W", "D", "W"]
        assert body["rates"]["btts"] == 0.6
        assert body["recent_matches"][0]["result"] == "D"

    def test_query_is_forwarded_to_the_repository(self):
        repo = AsyncMock()
        repo.get_team_stats.return_value = team_payload()
        client(repo).get("/api/stats/team", params={"league_code": "P1", "team": "Sporting"})
        query = repo.get_team_stats.await_args.args[0]
        assert query.league_code == "P1"
        assert query.team == "Sporting"

    def test_unknown_team_returns_404(self):
        repo = AsyncMock()
        repo.get_team_stats.side_effect = TeamNotFoundError("unknown team")
        res = client(repo).get("/api/stats/team", params={"league_code": "P1", "team": "Nobody"})
        assert res.status_code == 404

    def test_unavailable_upstream_returns_503(self):
        repo = AsyncMock()
        repo.get_team_stats.side_effect = StatsUnavailableError("space down")
        res = client(repo).get("/api/stats/team", params={"league_code": "P1", "team": "Sporting"})
        assert res.status_code == 503

    def test_missing_query_params_returns_422(self):
        repo = AsyncMock()
        res = client(repo).get("/api/stats/team", params={"league_code": "P1"})
        assert res.status_code == 422

    def test_unauthenticated_returns_401(self):
        app = FastAPI()
        app.include_router(router)
        res = TestClient(app, raise_server_exceptions=False).get(
            "/api/stats/team", params={"league_code": "P1", "team": "Sporting"}
        )
        assert res.status_code == 401


class TestMatchStatsEndpoint:

    @staticmethod
    def _body() -> dict:
        return {"home_team": "Sporting", "away_team": "Porto", "league_code": "P1"}

    def test_returns_match_payload(self):
        repo = AsyncMock()
        repo.get_match_stats.return_value = match_payload(GOAL_MARKETS)
        res = client(repo).post("/api/stats/match", json=self._body())
        assert res.status_code == 200
        body = res.json()
        assert body["head_to_head"]["played"] == 3
        assert body["home"]["team"] == "Sporting"
        assert body["away"]["team"] == "Porto"
        assert body["goal_markets"]["expected_goals"]["total"] == 2.9
        assert body["goal_markets"]["top_scorelines"][0]["score"] == "2-1"

    def test_markets_from_a_space_predating_the_source_field_still_validate(self):
        """Deploy ordering: the backend may lead the Space by a few minutes."""
        legacy = {k: v for k, v in GOAL_MARKETS.items() if k != "source"}
        repo = AsyncMock()
        repo.get_match_stats.return_value = match_payload(legacy)
        res = client(repo).post("/api/stats/match", json=self._body())
        assert res.status_code == 200
        assert res.json()["goal_markets"]["source"] == "model"

    def test_goal_markets_may_be_absent(self):
        repo = AsyncMock()
        repo.get_match_stats.return_value = match_payload(None)
        res = client(repo).post("/api/stats/match", json=self._body())
        assert res.status_code == 200
        assert res.json()["goal_markets"] is None

    def test_query_is_forwarded_to_the_repository(self):
        repo = AsyncMock()
        repo.get_match_stats.return_value = match_payload()
        client(repo).post("/api/stats/match", json=self._body())
        query = repo.get_match_stats.await_args.args[0]
        assert query.home_team == "Sporting"
        assert query.away_team == "Porto"
        assert query.league_code == "P1"

    def test_unknown_team_returns_404(self):
        repo = AsyncMock()
        repo.get_match_stats.side_effect = TeamNotFoundError("unknown team")
        res = client(repo).post("/api/stats/match", json=self._body())
        assert res.status_code == 404

    def test_unavailable_upstream_returns_503(self):
        repo = AsyncMock()
        repo.get_match_stats.side_effect = StatsUnavailableError("space down")
        res = client(repo).post("/api/stats/match", json=self._body())
        assert res.status_code == 503

    def test_incomplete_body_returns_422(self):
        repo = AsyncMock()
        res = client(repo).post("/api/stats/match", json={"home_team": "Sporting"})
        assert res.status_code == 422

    def test_unauthenticated_returns_401(self):
        app = FastAPI()
        app.include_router(router)
        res = TestClient(app, raise_server_exceptions=False).post(
            "/api/stats/match", json=self._body()
        )
        assert res.status_code == 401


class TestStatsRepositoryProvider:
    """The provider wires the Space-backed repository from injected config."""

    @staticmethod
    def _request() -> MagicMock:
        config = MagicMock()
        config.inference.enabled = True
        config.inference.space_url = "https://space.test"
        request = MagicMock()
        request.app.state.config = config
        return request

    def test_provides_a_remote_repository(self):
        repo = stats_module.get_stats_repository(self._request())
        assert isinstance(repo, RemoteStatsRepository)

    @pytest.mark.asyncio
    async def test_repository_delegates_to_the_inference_service(self, monkeypatch):
        from src.backend.services.inference_service import InferenceService

        monkeypatch.setattr(
            InferenceService,
            "get_team_insights",
            AsyncMock(return_value=team_payload()),
        )
        repo = stats_module.get_stats_repository(self._request())
        from src.backend.repositories.stats_repository import TeamStatsQuery

        result = await repo.get_team_stats(
            TeamStatsQuery(league_code="P1", team="Sporting")
        )
        assert result["team"] == "Sporting"
