"""Tests for POST /api/predictions/custom endpoint."""

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.inference import router, get_inference_service
from src.backend.core.auth import get_current_user
from src.backend.services.inference_service import PredictionRefused


def _make_client(mock_service: object, user_id: str = "user-123") -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: user_id
    app.dependency_overrides[get_inference_service] = lambda: mock_service
    return TestClient(app, raise_server_exceptions=False)


class TestCustomPredictEndpoint:

    def test_returns_200_with_prediction(self):
        mock_svc = MagicMock()
        mock_svc.predict_custom = AsyncMock()
        mock_svc.predict_custom.return_value = {
            "home_team": "Sporting CP",
            "away_team": "Tondela",
            "predicted_outcome": "Home Win",
            "confidence": 0.72,
            "probabilities": {"home_win": 0.72, "draw": 0.18, "away_win": 0.10},
            "league": "Liga Portugal",
        }
        res = _make_client(mock_svc).post(
            "/api/predictions/custom",
            json={"home_team": "Sporting CP", "away_team": "Tondela", "league_code": "P1"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["home_team"] == "Sporting CP"
        assert data["predicted_outcome"] == "Home Win"

    def test_passes_teams_to_service(self):
        mock_svc = MagicMock()
        mock_svc.predict_custom = AsyncMock()
        mock_svc.predict_custom.return_value = {
            "home_team": "Arsenal", "away_team": "Chelsea",
            "predicted_outcome": "Draw", "confidence": 0.4,
            "probabilities": {"home_win": 0.35, "draw": 0.4, "away_win": 0.25},
            "league": "Premier League",
        }
        _make_client(mock_svc).post(
            "/api/predictions/custom",
            json={"home_team": "Arsenal", "away_team": "Chelsea", "league_code": "E0"},
        )
        mock_svc.predict_custom.assert_called_once_with(
            home_team="Arsenal",
            away_team="Chelsea",
            league_code="E0",
            away_league_code=None,
        )

    def test_service_error_returns_503(self):
        mock_svc = MagicMock()
        mock_svc.predict_custom = AsyncMock()
        mock_svc.predict_custom.side_effect = RuntimeError("HF Space unavailable")
        res = _make_client(mock_svc).post(
            "/api/predictions/custom",
            json={"home_team": "A", "away_team": "B", "league_code": "E0"},
        )
        assert res.status_code == 503

    def test_unauthenticated_returns_401(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post(
            "/api/predictions/custom",
            json={"home_team": "A", "away_team": "B", "league_code": "E0"},
        )
        assert res.status_code == 401

    def test_missing_body_returns_422(self):
        mock_svc = MagicMock()
        res = _make_client(mock_svc).post("/api/predictions/custom", json={})
        assert res.status_code == 422


class TestCrossLeagueRequest:
    """A fixture whose two sides come from different domestic leagues."""

    @staticmethod
    def _service(result: dict | None = None) -> MagicMock:
        mock_svc = MagicMock()
        mock_svc.predict_custom = AsyncMock()
        mock_svc.predict_custom.return_value = result or {
            "home_team": "Arsenal",
            "away_team": "Benfica",
            "predicted_outcome": "Home Win",
            "confidence": 0.58,
            "probabilities": {"home_win": 0.58, "draw": 0.22, "away_win": 0.20},
            "league": "Premier League",
            "away_league": "Liga Portugal",
            "model": "elo",
        }
        return mock_svc

    def test_away_league_reaches_the_service(self):
        mock_svc = self._service()
        _make_client(mock_svc).post(
            "/api/predictions/custom",
            json={
                "home_team": "Arsenal",
                "away_team": "Benfica",
                "league_code": "E0",
                "away_league_code": "P1",
            },
        )
        mock_svc.predict_custom.assert_called_once_with(
            home_team="Arsenal",
            away_team="Benfica",
            league_code="E0",
            away_league_code="P1",
        )

    def test_model_and_away_league_reach_the_client(self):
        """Both are the whole point: without them the UI cannot say the
        ensemble was bypassed, and would present an ELO number as one."""
        res = _make_client(self._service()).post(
            "/api/predictions/custom",
            json={
                "home_team": "Arsenal",
                "away_team": "Benfica",
                "league_code": "E0",
                "away_league_code": "P1",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["model"] == "elo"
        assert body["away_league"] == "Liga Portugal"

    def test_a_space_without_the_fields_still_answers(self):
        """The Space deploys separately. An older one omits both keys, and a
        domestic prediction must not start failing while it catches up."""
        legacy = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "predicted_outcome": "Draw",
            "confidence": 0.4,
            "probabilities": {"home_win": 0.35, "draw": 0.4, "away_win": 0.25},
            "league": "Premier League",
        }
        res = _make_client(self._service(legacy)).post(
            "/api/predictions/custom",
            json={"home_team": "Arsenal", "away_team": "Chelsea", "league_code": "E0"},
        )
        assert res.status_code == 200
        assert res.json()["model"] is None
        assert res.json()["away_league"] is None

    def test_refusal_returns_422_with_the_reason(self):
        """Not 503: the service is up and has answered. Reporting an outage
        would send the user to retry something that will never succeed."""
        mock_svc = MagicMock()
        mock_svc.predict_custom = AsyncMock()
        mock_svc.predict_custom.side_effect = PredictionRefused(
            "no history for 'Kairat' — its league is not tracked"
        )
        res = _make_client(mock_svc).post(
            "/api/predictions/custom",
            json={
                "home_team": "Arsenal",
                "away_team": "Kairat",
                "league_code": "E0",
                "away_league_code": "P1",
            },
        )
        assert res.status_code == 422
        assert "Kairat" in res.json()["detail"]
