"""Payload parity between the calculator and the API response contract.

The numbers are produced in the HF Space by ``TeamInsightsCalculator`` and
validated in the backend by the ``/api/stats`` response models. The two halves
live in different deployment units and are exercised by different tests, so a
renamed field would pass both suites and only fail in production.

These tests feed real calculator output through the real response models —
the one place the two shapes are compared directly.
"""

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from config.config_loader import InsightsConfig
from src.analysis.team_insights import (
    FixtureQuery,
    TeamInsightsCalculator,
    TeamQuery,
)
from src.backend.api.stats import (
    MatchStatsResponse,
    TeamStatsResponse,
    get_stats_repository,
    router,
)
from src.backend.core.auth import get_current_user

LEAGUES = {"P1": "Liga Portugal"}
CONFIG = InsightsConfig(
    recent_matches=5, h2h_matches=5, form_sequence_length=5, max_scorelines=3
)


class StubPrediction:
    lambda_home = 1.7
    lambda_away = 1.2
    over_15 = 0.75
    over_25 = 0.53
    over_35 = 0.30
    under_25 = 0.47
    btts_yes = 0.55
    btts_no = 0.45
    top_scorelines = [(2, 1, 0.11), (1, 1, 0.10)]


class StubMarketModel:
    def knows(self, team: str) -> bool:
        return True

    def predict(self, home_team: str, away_team: str) -> StubPrediction:
        return StubPrediction()


def calculator(market_model: object | None = None) -> TeamInsightsCalculator:
    frame = pd.DataFrame(
        [
            ("2024-01-01", "Sporting", "Porto", 2, 1, 10, 9, 5, 4, 6, 3, 1, 2, 0, 0),
            ("2024-01-08", "Porto", "Sporting", 0, 0, 11, 8, 4, 3, 5, 4, 1, 2, 0, 0),
            ("2024-01-15", "Sporting", "Benfica", 3, 1, 14, 7, 7, 2, 8, 3, 1, 1, 0, 0),
        ],
        columns=[
            "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST",
            "AST", "HC", "AC", "HY", "AY", "HR", "AR",
        ],
    )
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["League"] = LEAGUES["P1"]
    return TeamInsightsCalculator(
        matches=frame, config=CONFIG, leagues=LEAGUES, market_model=market_model
    )


def team_payload() -> dict:
    return (
        calculator()
        .team_insights(TeamQuery(league_code="P1", team="Sporting"))
        .to_dict()
    )


def match_payload(market_model: object | None = None) -> dict:
    return (
        calculator(market_model)
        .match_insights(
            FixtureQuery(league_code="P1", home_team="Sporting", away_team="Porto")
        )
        .to_dict()
    )


class TestResponseModelsAcceptCalculatorOutput:

    def test_team_payload_validates(self):
        response = TeamStatsResponse(**team_payload())
        assert response.team == "Sporting"
        assert response.overall.played == 3

    def test_match_payload_validates_without_markets(self):
        response = MatchStatsResponse(**match_payload())
        assert response.goal_markets is None
        assert response.head_to_head.played == 2

    def test_match_payload_validates_with_markets(self):
        response = MatchStatsResponse(**match_payload(StubMarketModel()))
        assert response.goal_markets is not None
        assert response.goal_markets.expected_goals.total == pytest.approx(2.9)
        assert response.goal_markets.top_scorelines[0].score == "2-1"

    def test_no_field_is_dropped_in_translation(self):
        """Extra keys upstream are fine; missing ones are not."""
        payload = team_payload()
        response = TeamStatsResponse(**payload).model_dump()
        assert set(payload) == set(response)
        assert set(payload["overall"]) == set(response["overall"])
        assert set(payload["rates"]) == set(response["rates"])
        assert set(payload["averages"]) == set(response["averages"])
        assert set(payload["recent_matches"][0]) == set(response["recent_matches"][0])


class TestEndToEndThroughTheRouter:
    """Calculator output, carried by a stubbed repository, served by the API."""

    @staticmethod
    def _client(payload: dict) -> TestClient:
        repo = AsyncMock()
        repo.get_team_stats.return_value = payload
        repo.get_match_stats.return_value = payload
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: "user-123"
        app.dependency_overrides[get_stats_repository] = lambda: repo
        return TestClient(app, raise_server_exceptions=False)

    def test_team_endpoint_serves_calculator_output(self):
        res = self._client(team_payload()).get(
            "/api/stats/team", params={"league_code": "P1", "team": "Sporting"}
        )
        assert res.status_code == 200
        body = res.json()
        assert (body["overall"]["wins"], body["overall"]["draws"]) == (2, 1)
        assert body["form_sequence"] == ["W", "D", "W"]
        assert body["recent_matches"][0]["date"] == "2024-01-15"

    def test_match_endpoint_serves_calculator_output(self):
        res = self._client(match_payload(StubMarketModel())).post(
            "/api/stats/match",
            json={"home_team": "Sporting", "away_team": "Porto", "league_code": "P1"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["head_to_head"]["home_wins"] == 1
        assert body["head_to_head"]["draws"] == 1
        assert body["goal_markets"]["btts"]["yes"] == 0.55
