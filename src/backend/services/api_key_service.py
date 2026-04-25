"""CRUD service for user Gemini API keys stored encrypted in Supabase."""

from typing import Optional

from supabase import Client

from src.backend.core.encryption import EncryptionService

_TABLE = "user_api_keys"


class ApiKeyService:
    def __init__(self, supabase: Client, encryption: EncryptionService) -> None:
        self._db = supabase
        self._enc = encryption

    def get_user_key(self, user_id: str) -> Optional[str]:
        response = (
            self._db.table(_TABLE)
            .select("encrypted_gemini_key")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        data: Optional[dict] = response.data  # type: ignore[union-attr,assignment]
        if data is None:
            return None
        return self._enc.decrypt(str(data["encrypted_gemini_key"]))

    def set_user_key(self, user_id: str, plaintext_key: str) -> None:
        if not plaintext_key:
            raise ValueError("Gemini key cannot be empty")
        encrypted = self._enc.encrypt(plaintext_key)
        self._db.table(_TABLE).upsert(
            {"user_id": user_id, "encrypted_gemini_key": encrypted},
            on_conflict="user_id",
        ).execute()

    def delete_user_key(self, user_id: str) -> None:
        self._db.table(_TABLE).delete().eq("user_id", user_id).execute()
