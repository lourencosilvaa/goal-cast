"""CRUD service for user Gemini API keys stored encrypted in Supabase."""

from typing import Optional

from supabase import Client

from src.backend.core.encryption import EncryptionService

_TABLE = "user_api_keys"
_SERVICE_GEMINI = "gemini"


class ApiKeyService:
    def __init__(self, supabase: Client, encryption: EncryptionService) -> None:
        self._db = supabase
        self._enc = encryption

    def get_user_key(self, user_id: str) -> Optional[str]:
        try:
            response = (
                self._db.table(_TABLE)
                .select("key_enc")
                .eq("user_id", user_id)
                .eq("service", _SERVICE_GEMINI)
                .maybe_single()
                .execute()
            )
        except Exception:
            return None
        if response is None:
            return None
        data: Optional[dict] = response.data  # type: ignore[union-attr,assignment]
        if data is None:
            return None
        return self._enc.decrypt(str(data["key_enc"]))

    def set_user_key(self, user_id: str, plaintext_key: str) -> None:
        if not plaintext_key:
            raise ValueError("Gemini key cannot be empty")
        encrypted = self._enc.encrypt(plaintext_key)
        self._db.table(_TABLE).upsert(
            {
                "user_id": user_id,
                "service": _SERVICE_GEMINI,
                "key_enc": encrypted,
            },
            on_conflict="user_id",
        ).execute()

    def delete_user_key(self, user_id: str) -> None:
        self._db.table(_TABLE).delete().eq("user_id", user_id).eq(
            "service", _SERVICE_GEMINI
        ).execute()
