"""Tests for the /api/predictions/infer endpoint."""

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


class TestInferenceEndpoint:

    def test_post_returns_200_with_results(self):
        mock_svc = MagicMock()
        mock_svc.run = AsyncMock()
        mock_svc.run.return_value = [
            {"home_team": "Arsenal", "away_team": "Chelsea",
             "predicted_outcome": "Home Win", "confidence": 0.65}
        ]
        res = _make_client(mock_svc).post(
            "/api/predictions/infer",
            params={"date": "28/04/2024", "league_code": "E0"},
        )
        assert res.status_code == 200

    def test_post_returns_list_in_predictions_key(self):
        mock_svc = MagicMock()
        mock_svc.run = AsyncMock()
        mock_svc.run.return_value = [
            {"home_team": "Arsenal", "away_team": "Chelsea"}
        ]
        res = _make_client(mock_svc).post(
            "/api/predictions/infer",
            params={"date": "28/04/2024"},
        )
        assert "predictions" in res.json()
        assert isinstance(res.json()["predictions"], list)

    def test_post_empty_fixtures_returns_empty_list(self):
        mock_svc = MagicMock()
        mock_svc.run = AsyncMock()
        mock_svc.run.return_value = []
        res = _make_client(mock_svc).post(
            "/api/predictions/infer",
            params={"date": "28/04/2024"},
        )
        assert res.json()["predictions"] == []

    def test_post_disabled_returns_503(self):
        mock_svc = MagicMock()
        mock_svc.run = AsyncMock()
        mock_svc.run.side_effect = RuntimeError("inference disabled")
        res = _make_client(mock_svc).post(
            "/api/predictions/infer",
            params={"date": "28/04/2024"},
        )
        assert res.status_code == 503

    def test_unauthenticated_returns_401(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post("/api/predictions/infer", params={"date": "28/04/2024"})
        assert res.status_code == 401

    def test_passes_league_code_to_service(self):
        mock_svc = MagicMock()
        mock_svc.run = AsyncMock()
        mock_svc.run.return_value = []
        _make_client(mock_svc).post(
            "/api/predictions/infer",
            params={"date": "28/04/2024", "league_code": "SP1"},
        )
        call_kwargs = mock_svc.run.call_args
        assert "SP1" in str(call_kwargs)
