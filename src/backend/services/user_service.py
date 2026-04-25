"""CRUD service for user_profiles in Supabase."""

from typing import Optional

from supabase import Client

_TABLE = "user_profiles"


class UserService:
    def __init__(self, supabase: Client) -> None:
        self._db = supabase

    def get_profile(self, user_id: str) -> Optional[dict]:
        response = (
            self._db.table(_TABLE)
            .select("*")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        if response is None:
            return None
        data: Optional[dict] = response.data  # type: ignore[union-attr,assignment]
        return data

    def list_users(self) -> list[dict]:
        response = (
            self._db.table(_TABLE).select("*").order("created_at", desc=False).execute()
        )
        data: list[dict] = response.data or []  # type: ignore[assignment]
        return data

    def create_profile(self, user_id: str, email: str, approved: bool = True) -> None:
        self._db.table(_TABLE).insert(
            {
                "user_id": user_id,
                "email": email,
                "approved": approved,
                "is_admin": False,
            }
        ).execute()

    def set_approved(self, user_id: str, approved: bool) -> None:
        self._db.table(_TABLE).update({"approved": approved}).eq(
            "user_id", user_id
        ).execute()
