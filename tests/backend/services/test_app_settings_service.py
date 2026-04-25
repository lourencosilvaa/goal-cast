"""Tests for AppSettingsService."""

from unittest.mock import MagicMock

from src.backend.services.app_settings_service import AppSettingsService


class TestAppSettingsService:
    def _make_service(self, row_data=None) -> tuple[AppSettingsService, MagicMock]:
        mock_db = MagicMock()
        mock_db.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = row_data
        svc = AppSettingsService(supabase=mock_db)
        return svc, mock_db

    def test_get_returns_none_when_key_not_found(self) -> None:
        svc, _ = self._make_service(row_data=None)
        assert svc.get("retraining") is None

    def test_get_returns_value_when_key_exists(self) -> None:
        svc, _ = self._make_service(row_data={"key": "retraining", "value": "true"})
        assert svc.get("retraining") == "true"

    def test_set_upserts_row(self) -> None:
        svc, mock_db = self._make_service()
        svc.set("retraining", "true")
        mock_db.table.return_value.upsert.assert_called_once_with(
            {"key": "retraining", "value": "true"}
        )
