"""Tests for /api/keys routes."""

import pytest
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


class TestGetGeminiKey:
    def test_returns_has_key_true_when_key_exists(self) -> None:
        mock_service = MagicMock()
        mock_service.get_user_key.return_value = "AIzaSy_existing_key"
        res = _make_client(mock_service).get("/api/keys/gemini")
        assert res.status_code == 200
        assert res.json()["has_key"] is True

    def test_returns_has_key_false_when_no_key(self) -> None:
        mock_service = MagicMock()
        mock_service.get_user_key.return_value = None
        res = _make_client(mock_service).get("/api/keys/gemini")
        assert res.status_code == 200
        assert res.json()["has_key"] is False

    def test_unauthenticated_returns_401(self) -> None:
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/api/keys/gemini")
        assert res.status_code == 401


class TestPutGeminiKey:
    def test_saves_key_successfully(self) -> None:
        mock_service = MagicMock()
        res = _make_client(mock_service).put(
            "/api/keys/gemini", json={"key": "AIzaSy_brand_new_key"}
        )
        assert res.status_code == 200
        assert res.json()["success"] is True

    def test_empty_key_returns_400(self) -> None:
        mock_service = MagicMock()
        mock_service.set_user_key.side_effect = ValueError("key cannot be empty")
        res = _make_client(mock_service).put("/api/keys/gemini", json={"key": ""})
        assert res.status_code == 400

    # suppress unused import warning — pytest requires the marker
    @pytest.mark.usefixtures()
    def test_delete_key_successfully(self) -> None:
        mock_service = MagicMock()
        res = _make_client(mock_service).delete("/api/keys/gemini")
        assert res.status_code == 200
        assert res.json()["success"] is True
