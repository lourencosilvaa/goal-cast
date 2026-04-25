"""Service for reading and writing generic app settings in Supabase."""

from typing import Optional

from supabase import Client

_TABLE = "app_settings"


class AppSettingsService:
    def __init__(self, supabase: Client) -> None:
        self._db = supabase

    def get(self, key: str) -> Optional[str]:
        response = (
            self._db.table(_TABLE).select("*").eq("key", key).maybe_single().execute()
        )
        if response is None:
            return None
        row: Optional[dict] = response.data  # type: ignore[union-attr,assignment]
        return row["value"] if row else None

    def set(self, key: str, value: str) -> None:
        self._db.table(_TABLE).upsert({"key": key, "value": value}).execute()
