"""CRUD service for user API keys stored encrypted in Supabase."""

from typing import Optional

from src.backend.core.encryption import EncryptionService
from supabase import Client

_TABLE = "user_api_keys"
_SERVICE_GEMINI = "gemini"
_SERVICE_NVIDIA = "nvidia"


class ApiKeyService:
    def __init__(self, supabase: Client, encryption: EncryptionService) -> None:
        self._db = supabase
        self._enc = encryption

    def get_user_key(
        self, user_id: str, service: str = _SERVICE_GEMINI
    ) -> Optional[str]:
        try:
            response = (
                self._db.table(_TABLE)
                .select("key_enc")
                .eq("user_id", user_id)
                .eq("service", service)
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

    def set_user_key(
        self, user_id: str, plaintext_key: str, service: str = _SERVICE_GEMINI
    ) -> None:
        if not plaintext_key:
            raise ValueError(f"{service} key cannot be empty")
        encrypted = self._enc.encrypt(plaintext_key)
        self._db.table(_TABLE).upsert(
            {
                "user_id": user_id,
                "service": service,
                "key_enc": encrypted,
            },
            on_conflict="user_id,service",
        ).execute()

    def delete_user_key(self, user_id: str, service: str = _SERVICE_GEMINI) -> None:
        self._db.table(_TABLE).delete().eq("user_id", user_id).eq(
            "service", service
        ).execute()
