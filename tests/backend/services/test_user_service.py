"""Phase 1 tests for UserService — must FAIL before implementation."""

import pytest
from unittest.mock import MagicMock
from src.backend.services.user_service import UserService


def _make_service(mock_client: MagicMock) -> UserService:
    return UserService(supabase=mock_client)


class TestGetUserProfile:
    def test_returns_profile_when_found(self) -> None:
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
            data={"user_id": "u1", "email": "a@b.com", "approved": True, "is_admin": False}
        )
        svc = _make_service(mock_client)
        result = svc.get_profile(user_id="u1")
        assert result is not None
        assert result["email"] == "a@b.com"
        assert result["approved"] is True

    def test_returns_none_when_not_found(self) -> None:
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
            data=None
        )
        svc = _make_service(mock_client)
        assert svc.get_profile(user_id="missing") is None


class TestListUsers:
    def test_returns_all_users(self) -> None:
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.order.return_value.execute.return_value = MagicMock(
            data=[
                {"user_id": "u1", "email": "a@b.com", "approved": True, "is_admin": True},
                {"user_id": "u2", "email": "c@d.com", "approved": False, "is_admin": False},
            ]
        )
        svc = _make_service(mock_client)
        users = svc.list_users()
        assert len(users) == 2
        assert users[0]["email"] == "a@b.com"


class TestSetApproved:
    def test_updates_approved_column(self) -> None:
        mock_client = MagicMock()
        svc = _make_service(mock_client)
        svc.set_approved(user_id="u2", approved=True)
        mock_client.table.assert_called_with("user_profiles")
        mock_client.table.return_value.update.assert_called_once_with({"approved": True})

    def test_can_revoke_access(self) -> None:
        mock_client = MagicMock()
        svc = _make_service(mock_client)
        svc.set_approved(user_id="u2", approved=False)
        mock_client.table.return_value.update.assert_called_once_with({"approved": False})


class TestCreateUserProfile:
    def test_inserts_approved_profile(self) -> None:
        mock_client = MagicMock()
        svc = _make_service(mock_client)
        svc.create_profile(user_id="u3", email="new@user.com", approved=True)
        mock_client.table.assert_called_with("user_profiles")
        call_data = mock_client.table.return_value.insert.call_args[0][0]
        assert call_data["user_id"] == "u3"
        assert call_data["email"] == "new@user.com"
        assert call_data["approved"] is True
        assert call_data["is_admin"] is False
