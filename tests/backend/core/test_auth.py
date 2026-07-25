"""Tests for JWT auth dependency."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException


class TestValidateToken:
    def test_valid_token_returns_user_id(self) -> None:
        from src.backend.core.auth import _validate_token

        mock_client = MagicMock()
        mock_client.auth.get_user.return_value = MagicMock(user=MagicMock(id="user-123"))

        with patch("src.backend.core.auth.get_supabase_client", return_value=mock_client):
            user_id = _validate_token("valid.jwt.token")

        assert user_id == "user-123"

    def test_invalid_token_raises_401(self) -> None:
        from src.backend.core.auth import _validate_token

        mock_client = MagicMock()
        mock_client.auth.get_user.return_value = MagicMock(user=None)

        with patch("src.backend.core.auth.get_supabase_client", return_value=mock_client):
            with pytest.raises(HTTPException) as exc:
                _validate_token("bad.token")

        assert exc.value.status_code == 401

    def test_supabase_error_raises_401(self) -> None:
        from src.backend.core.auth import _validate_token

        mock_client = MagicMock()
        mock_client.auth.get_user.side_effect = Exception("network error")

        with patch("src.backend.core.auth.get_supabase_client", return_value=mock_client):
            with pytest.raises(HTTPException) as exc:
                _validate_token("any.token")

        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_credentials_raises_401(self) -> None:
        from src.backend.core.auth import get_current_user

        with pytest.raises(HTTPException) as exc:
            await get_current_user(credentials=None)

        assert exc.value.status_code == 401


class TestDevBypass:
    @pytest.mark.asyncio
    async def test_get_current_user_returns_dev_id_when_enabled(self, monkeypatch) -> None:
        from src.backend.core.auth import get_current_user

        monkeypatch.setenv("DEV_AUTH_BYPASS", "true")
        monkeypatch.setenv("DEV_USER_ID", "local-dev")

        # No credentials and no Supabase call needed.
        user_id = await get_current_user(credentials=None)

        assert user_id == "local-dev"

    @pytest.mark.asyncio
    async def test_get_approved_user_skips_profile_check_when_enabled(self, monkeypatch) -> None:
        from src.backend.core.auth import get_approved_user

        monkeypatch.setenv("DEV_AUTH_BYPASS", "true")
        monkeypatch.setenv("DEV_USER_ID", "local-dev")

        # Would raise if Supabase were touched (no client configured in tests).
        user_id = await get_approved_user(user_id="local-dev")

        assert user_id == "local-dev"

    @pytest.mark.asyncio
    async def test_disabled_bypass_still_requires_credentials(self, monkeypatch) -> None:
        from src.backend.core.auth import get_current_user

        monkeypatch.setenv("DEV_AUTH_BYPASS", "false")

        with pytest.raises(HTTPException) as exc:
            await get_current_user(credentials=None)

        assert exc.value.status_code == 401

