"""Tests for /api/retrain-status routes."""

import os
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.status import router, get_app_settings_service


def _make_client(mock_service: object) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_app_settings_service] = lambda: mock_service
    return TestClient(app, raise_server_exceptions=False)


class TestGetRetrainStatus:
    def test_returns_false_when_not_set(self) -> None:
        mock_service = MagicMock()
        mock_service.get.return_value = None
        res = _make_client(mock_service).get("/api/retrain-status")
        assert res.status_code == 200
        assert res.json() == {"retraining": False}

    def test_returns_true_when_flag_is_true(self) -> None:
        mock_service = MagicMock()
        mock_service.get.return_value = "true"
        res = _make_client(mock_service).get("/api/retrain-status")
        assert res.status_code == 200
        assert res.json() == {"retraining": True}

    def test_returns_false_when_flag_is_false(self) -> None:
        mock_service = MagicMock()
        mock_service.get.return_value = "false"
        res = _make_client(mock_service).get("/api/retrain-status")
        assert res.status_code == 200
        assert res.json() == {"retraining": False}


class TestSetRetrainStatus:
    def test_missing_api_key_returns_401(self) -> None:
        mock_service = MagicMock()
        with patch.dict(os.environ, {"RETRAIN_API_KEY": "secret123"}):
            res = _make_client(mock_service).post(
                "/api/retrain-status", json={"retraining": True}
            )
        assert res.status_code == 401

    def test_wrong_api_key_returns_401(self) -> None:
        mock_service = MagicMock()
        with patch.dict(os.environ, {"RETRAIN_API_KEY": "secret123"}):
            res = _make_client(mock_service).post(
                "/api/retrain-status",
                json={"retraining": True},
                headers={"X-Api-Key": "wrong-key"},
            )
        assert res.status_code == 401

    def test_valid_key_sets_flag_true(self) -> None:
        mock_service = MagicMock()
        with patch.dict(os.environ, {"RETRAIN_API_KEY": "secret123"}):
            res = _make_client(mock_service).post(
                "/api/retrain-status",
                json={"retraining": True},
                headers={"X-Api-Key": "secret123"},
            )
        assert res.status_code == 200
        mock_service.set.assert_called_once_with("retraining", "true")

    def test_valid_key_sets_flag_false(self) -> None:
        mock_service = MagicMock()
        with patch.dict(os.environ, {"RETRAIN_API_KEY": "secret123"}):
            res = _make_client(mock_service).post(
                "/api/retrain-status",
                json={"retraining": False},
                headers={"X-Api-Key": "secret123"},
            )
        assert res.status_code == 200
        mock_service.set.assert_called_once_with("retraining", "false")

    def test_no_env_var_configured_returns_503(self) -> None:
        mock_service = MagicMock()
        env_without_key = {k: v for k, v in os.environ.items() if k != "RETRAIN_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            res = _make_client(mock_service).post(
                "/api/retrain-status",
                json={"retraining": True},
                headers={"X-Api-Key": "anything"},
            )
        assert res.status_code == 503

    def test_missing_body_returns_422(self) -> None:
        mock_service = MagicMock()
        with patch.dict(os.environ, {"RETRAIN_API_KEY": "secret123"}):
            res = _make_client(mock_service).post(
                "/api/retrain-status",
                json={},
                headers={"X-Api-Key": "secret123"},
            )
        assert res.status_code == 422
