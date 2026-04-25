"""Tests for ApiKeyService."""

import pytest
from unittest.mock import MagicMock
from src.backend.core.encryption import EncryptionService


class TestApiKeyService:
    def setup_method(self) -> None:
        self.fernet_key = EncryptionService.generate_key()
        self.encryption = EncryptionService(fernet_key=self.fernet_key)

    def _make_service(self, mock_client: MagicMock) -> object:
        from src.backend.services.api_key_service import ApiKeyService
        return ApiKeyService(supabase=mock_client, encryption=self.encryption)

    def test_get_user_key_returns_decrypted_key(self) -> None:
        encrypted = self.encryption.encrypt("AIzaSy_real_key")
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
            data={"encrypted_gemini_key": encrypted}
        )

        service = self._make_service(mock_client)
        result = service.get_user_key(user_id="user-123")

        assert result == "AIzaSy_real_key"

    def test_get_user_key_returns_none_when_not_found(self) -> None:
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
            data=None
        )

        service = self._make_service(mock_client)
        result = service.get_user_key(user_id="user-999")

        assert result is None

    def test_set_user_key_stores_encrypted(self) -> None:
        mock_client = MagicMock()

        service = self._make_service(mock_client)
        service.set_user_key(user_id="user-123", plaintext_key="AIzaSy_new_key")

        mock_client.table.assert_called_with("user_api_keys")
        call_args = mock_client.table.return_value.upsert.call_args[0][0]
        assert call_args["user_id"] == "user-123"
        assert "encrypted_gemini_key" in call_args
        assert call_args["encrypted_gemini_key"] != "AIzaSy_new_key"
        assert self.encryption.decrypt(call_args["encrypted_gemini_key"]) == "AIzaSy_new_key"

    def test_set_user_key_empty_string_raises(self) -> None:
        mock_client = MagicMock()
        service = self._make_service(mock_client)

        with pytest.raises(ValueError, match="key"):
            service.set_user_key(user_id="user-123", plaintext_key="")

    def test_delete_user_key(self) -> None:
        mock_client = MagicMock()
        service = self._make_service(mock_client)
        service.delete_user_key(user_id="user-123")
        mock_client.table.assert_called_with("user_api_keys")
