"""Tests for /api/admin routes."""

from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.admin import router, get_user_service, get_admin_user
from src.backend.core.auth import get_current_user


def _make_client(
    mock_service: object,
    is_admin: bool = True,
    user_id: str = "admin-user-id",
) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: user_id
    app.dependency_overrides[get_user_service] = lambda: mock_service
    if is_admin:
        app.dependency_overrides[get_admin_user] = lambda: user_id
    return TestClient(app, raise_server_exceptions=False)


class TestListUsers:
    def test_admin_can_list_users(self) -> None:
        mock_service = MagicMock()
        mock_service.list_users.return_value = [
            {"user_id": "u1", "email": "a@b.com", "approved": True, "is_admin": True},
            {"user_id": "u2", "email": "c@d.com", "approved": False, "is_admin": False},
        ]
        res = _make_client(mock_service, is_admin=True).get("/api/admin/users")
        assert res.status_code == 200
        assert len(res.json()) == 2

    def test_non_admin_gets_403(self) -> None:
        mock_service = MagicMock()
        mock_service.get_profile.return_value = {
            "user_id": "u2", "email": "c@d.com", "approved": True, "is_admin": False
        }
        # is_admin=False → get_admin_user NOT overridden → real dependency runs
        res = _make_client(mock_service, is_admin=False).get("/api/admin/users")
        assert res.status_code == 403


class TestCreateUser:
    def test_admin_creates_user(self) -> None:
        mock_service = MagicMock()
        mock_supabase = MagicMock()
        mock_supabase.auth.admin.create_user.return_value = MagicMock(
            user=MagicMock(id="new-uid")
        )

        with patch("src.backend.api.admin.get_supabase_client", return_value=mock_supabase):
            res = _make_client(mock_service, is_admin=True).post(
                "/api/admin/users",
                json={"email": "new@user.com", "password": "secret123"},
            )

        assert res.status_code == 201
        assert res.json()["email"] == "new@user.com"
        assert res.json()["approved"] is True

    def test_create_user_supabase_error_returns_400(self) -> None:
        mock_service = MagicMock()
        mock_supabase = MagicMock()
        mock_supabase.auth.admin.create_user.side_effect = Exception("already exists")

        with patch("src.backend.api.admin.get_supabase_client", return_value=mock_supabase):
            res = _make_client(mock_service, is_admin=True).post(
                "/api/admin/users",
                json={"email": "dup@user.com", "password": "secret123"},
            )

        assert res.status_code == 400


class TestPatchUser:
    def test_admin_can_approve_user(self) -> None:
        mock_service = MagicMock()
        res = _make_client(mock_service, is_admin=True).patch(
            "/api/admin/users/u2", json={"approved": True}
        )
        assert res.status_code == 200
        assert res.json()["success"] is True

    def test_admin_can_revoke_user(self) -> None:
        mock_service = MagicMock()
        res = _make_client(mock_service, is_admin=True).patch(
            "/api/admin/users/u2", json={"approved": False}
        )
        assert res.status_code == 200
        mock_service.set_approved.assert_called_once_with(user_id="u2", approved=False)
