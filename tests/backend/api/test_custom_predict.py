"""Tests for POST /api/predictions/custom endpoint."""

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.inference import router, get_inference_service
from src.backend.core.auth import get_current_user


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
            home_team="Arsenal", away_team="Chelsea", league_code="E0"
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
