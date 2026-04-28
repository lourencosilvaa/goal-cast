"""Tests for /api/keys/nvidia endpoints."""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.keys import router, get_api_key_service
from src.backend.core.auth import get_current_user


def _make_client(mock_service: object, user_id: str = "user-123") -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: user_id
    app.dependency_overrides[get_api_key_service] = lambda: mock_service
    return TestClient(app, raise_server_exceptions=False)


class TestGetNvidiaKey:

    def test_returns_has_key_true_when_key_exists(self):
        mock_service = MagicMock()
        mock_service.get_user_key.return_value = "nvapi-existing-key"
        res = _make_client(mock_service).get("/api/keys/nvidia")
        assert res.status_code == 200
        assert res.json()["has_key"] is True

    def test_returns_has_key_false_when_no_key(self):
        mock_service = MagicMock()
        mock_service.get_user_key.return_value = None
        res = _make_client(mock_service).get("/api/keys/nvidia")
        assert res.status_code == 200
        assert res.json()["has_key"] is False

    def test_unauthenticated_returns_401(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/api/keys/nvidia")
        assert res.status_code == 401

    def test_service_called_with_nvidia_service_name(self):
        mock_service = MagicMock()
        mock_service.get_user_key.return_value = None
        _make_client(mock_service).get("/api/keys/nvidia")
        mock_service.get_user_key.assert_called_once_with(
            user_id="user-123", service="nvidia"
        )


class TestPutNvidiaKey:

    def test_saves_key_successfully(self):
        mock_service = MagicMock()
        res = _make_client(mock_service).put(
            "/api/keys/nvidia", json={"key": "nvapi-brand-new-key"}
        )
        assert res.status_code == 200
        assert res.json()["success"] is True

    def test_empty_key_returns_400(self):
        mock_service = MagicMock()
        mock_service.set_user_key.side_effect = ValueError("key cannot be empty")
        res = _make_client(mock_service).put("/api/keys/nvidia", json={"key": ""})
        assert res.status_code == 400

    def test_service_called_with_nvidia_service_name(self):
        mock_service = MagicMock()
        _make_client(mock_service).put(
            "/api/keys/nvidia", json={"key": "nvapi-test"}
        )
        mock_service.set_user_key.assert_called_once_with(
            user_id="user-123", plaintext_key="nvapi-test", service="nvidia"
        )


class TestDeleteNvidiaKey:

    def test_delete_key_successfully(self):
        mock_service = MagicMock()
        res = _make_client(mock_service).delete("/api/keys/nvidia")
        assert res.status_code == 200
        assert res.json()["success"] is True

    def test_service_called_with_nvidia_service_name(self):
        mock_service = MagicMock()
        _make_client(mock_service).delete("/api/keys/nvidia")
        mock_service.delete_user_key.assert_called_once_with(
            user_id="user-123", service="nvidia"
        )
