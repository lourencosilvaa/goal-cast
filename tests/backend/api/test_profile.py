"""Tests for /api/users/me and /api/users/register routes."""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.profile import router, get_user_service
from src.backend.core.auth import get_current_user


def _make_client(
    mock_service: object,
    user_id: str = "test-user-id",
    authenticated: bool = True,
) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_user_service] = lambda: mock_service
    if authenticated:
        app.dependency_overrides[get_current_user] = lambda: user_id
    return TestClient(app, raise_server_exceptions=False)


class TestGetMyProfile:
    def test_returns_profile_for_authenticated_user(self) -> None:
        mock_service = MagicMock()
        mock_service.get_profile.return_value = {
            "user_id": "u1",
            "email": "me@test.com",
            "approved": True,
            "is_admin": False,
        }
        res = _make_client(mock_service, user_id="u1").get("/api/users/me")
        assert res.status_code == 200
        assert res.json()["email"] == "me@test.com"
        assert res.json()["approved"] is True

    def test_returns_404_when_profile_not_found(self) -> None:
        mock_service = MagicMock()
        mock_service.get_profile.return_value = None
        res = _make_client(mock_service).get("/api/users/me")
        assert res.status_code == 404

    def test_unauthenticated_returns_401(self) -> None:
        mock_service = MagicMock()
        res = _make_client(mock_service, authenticated=False).get("/api/users/me")
        assert res.status_code == 401


class TestRegisterUser:
    def test_register_creates_profile_with_approved_false(self) -> None:
        mock_service = MagicMock()
        mock_supabase = MagicMock()
        mock_supabase.auth.get_user.return_value = MagicMock(
            user=MagicMock(id="new-uid", email="new@user.com")
        )

        with patch("src.backend.api.profile.get_supabase_client", return_value=mock_supabase):
            res = _make_client(mock_service).post(
                "/api/users/register",
                json={"token": "valid-token"},
            )

        assert res.status_code == 201
        mock_service.create_profile.assert_called_once_with(
            user_id="new-uid",
            email="new@user.com",
            approved=False,
        )

    def test_register_invalid_token_returns_401(self) -> None:
        mock_service = MagicMock()
        mock_supabase = MagicMock()
        mock_supabase.auth.get_user.side_effect = Exception("invalid token")

        with patch("src.backend.api.profile.get_supabase_client", return_value=mock_supabase):
            res = _make_client(mock_service).post(
                "/api/users/register",
                json={"token": "bad-token"},
            )

        assert res.status_code == 401

    def test_register_without_token_or_user_id_returns_400(self) -> None:
        """Both fields are optional on the model, so an empty body is a valid
        request that the handler rejects with 400 — not a 422 schema error."""
        mock_service = MagicMock()
        with patch(
            "src.backend.api.profile.get_supabase_client", return_value=MagicMock()
        ):
            res = _make_client(mock_service).post("/api/users/register", json={})
        assert res.status_code == 400
        assert "token or user_id" in res.json()["detail"]

    def test_register_with_user_id_verifies_via_admin_api(self) -> None:
        """Email-confirmation-pending path: no token, but a known user_id."""
        mock_service = MagicMock()
        mock_supabase = MagicMock()
        mock_supabase.auth.admin.get_user_by_id.return_value = MagicMock(
            user=MagicMock(id="pending-uid", email="pending@user.com")
        )

        with patch(
            "src.backend.api.profile.get_supabase_client", return_value=mock_supabase
        ):
            res = _make_client(mock_service).post(
                "/api/users/register",
                json={"user_id": "pending-uid"},
            )

        assert res.status_code == 201
        mock_supabase.auth.admin.get_user_by_id.assert_called_once_with("pending-uid")

    def test_register_with_unknown_user_id_returns_401(self) -> None:
        mock_service = MagicMock()
        mock_supabase = MagicMock()
        mock_supabase.auth.admin.get_user_by_id.return_value = MagicMock(user=None)

        with patch(
            "src.backend.api.profile.get_supabase_client", return_value=mock_supabase
        ):
            res = _make_client(mock_service).post(
                "/api/users/register",
                json={"user_id": "ghost-uid"},
            )

        assert res.status_code == 401

    def test_register_when_profile_exists_still_returns_201(self) -> None:
        """Idempotent: re-registering (e.g. page refresh) should not crash."""
        mock_service = MagicMock()
        mock_service.create_profile.side_effect = Exception("duplicate key")
        mock_supabase = MagicMock()
        mock_supabase.auth.get_user.return_value = MagicMock(
            user=MagicMock(id="existing-uid", email="existing@user.com")
        )

        with patch("src.backend.api.profile.get_supabase_client", return_value=mock_supabase):
            res = _make_client(mock_service).post(
                "/api/users/register",
                json={"token": "valid-token"},
            )

        assert res.status_code == 201
