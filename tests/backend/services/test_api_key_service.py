"""Tests for ApiKeyService.

The service stores one encrypted key per (user_id, service) pair in the
``user_api_keys`` table, in a column named ``key_enc``. The Supabase query
chain it builds is ``table().select().eq().eq().maybe_single().execute()`` —
two ``eq`` calls, one for the user and one for the service — so the doubles
below mirror that chain exactly.
"""

from unittest.mock import MagicMock

import pytest

from src.backend.core.encryption import EncryptionService

_TABLE = "user_api_keys"


def _select_chain(mock_client: MagicMock) -> MagicMock:
    """The terminal ``execute`` of the service's select query."""
    return (
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute
    )


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
        _select_chain(mock_client).return_value = MagicMock(
            data={"key_enc": encrypted}
        )

        service = self._make_service(mock_client)

        assert service.get_user_key(user_id="user-123") == "AIzaSy_real_key"

    def test_get_user_key_returns_none_when_not_found(self) -> None:
        mock_client = MagicMock()
        _select_chain(mock_client).return_value = MagicMock(data=None)

        service = self._make_service(mock_client)

        assert service.get_user_key(user_id="user-999") is None

    def test_get_user_key_returns_none_when_response_is_none(self) -> None:
        mock_client = MagicMock()
        _select_chain(mock_client).return_value = None

        service = self._make_service(mock_client)

        assert service.get_user_key(user_id="user-999") is None

    def test_get_user_key_returns_none_when_query_raises(self) -> None:
        mock_client = MagicMock()
        _select_chain(mock_client).side_effect = RuntimeError("supabase down")

        service = self._make_service(mock_client)

        assert service.get_user_key(user_id="user-123") is None

    def test_get_user_key_filters_by_service(self) -> None:
        encrypted = self.encryption.encrypt("nvapi-key")
        mock_client = MagicMock()
        _select_chain(mock_client).return_value = MagicMock(
            data={"key_enc": encrypted}
        )

        service = self._make_service(mock_client)
        service.get_user_key(user_id="user-123", service="nvidia")

        second_eq = mock_client.table.return_value.select.return_value.eq.return_value.eq
        second_eq.assert_called_once_with("service", "nvidia")

    def test_set_user_key_stores_encrypted(self) -> None:
        mock_client = MagicMock()

        service = self._make_service(mock_client)
        service.set_user_key(user_id="user-123", plaintext_key="AIzaSy_new_key")

        mock_client.table.assert_called_with(_TABLE)
        payload = mock_client.table.return_value.upsert.call_args[0][0]
        assert payload["user_id"] == "user-123"
        assert payload["service"] == "gemini"
        assert payload["key_enc"] != "AIzaSy_new_key"
        assert self.encryption.decrypt(payload["key_enc"]) == "AIzaSy_new_key"

    def test_set_user_key_upserts_on_user_and_service(self) -> None:
        mock_client = MagicMock()

        service = self._make_service(mock_client)
        service.set_user_key(
            user_id="user-123", plaintext_key="nvapi-new", service="nvidia"
        )

        upsert_kwargs = mock_client.table.return_value.upsert.call_args[1]
        assert upsert_kwargs["on_conflict"] == "user_id,service"

    def test_set_user_key_empty_string_raises(self) -> None:
        mock_client = MagicMock()
        service = self._make_service(mock_client)

        with pytest.raises(ValueError, match="key"):
            service.set_user_key(user_id="user-123", plaintext_key="")

    def test_delete_user_key(self) -> None:
        mock_client = MagicMock()
        service = self._make_service(mock_client)
        service.delete_user_key(user_id="user-123")
        mock_client.table.assert_called_with(_TABLE)

    def test_delete_user_key_filters_by_user_and_service(self) -> None:
        mock_client = MagicMock()
        service = self._make_service(mock_client)
        service.delete_user_key(user_id="user-123", service="nvidia")

        delete_chain = mock_client.table.return_value.delete.return_value
        delete_chain.eq.assert_called_once_with("user_id", "user-123")
        delete_chain.eq.return_value.eq.assert_called_once_with("service", "nvidia")
