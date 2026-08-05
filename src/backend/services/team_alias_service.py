"""CRUD service for the ``team_aliases`` table in Supabase.

Holds the runtime half of canonical-name resolution: scraped names the
pipeline could not resolve (``pending``) and the mappings an admin has
confirmed (``approved``). Only approved rows are ever used to resolve a name —
see :mod:`src.teams.resolver` for why nothing is matched automatically.

Read paths degrade to an empty result so a page never fails because the alias
table is unavailable; **write** paths raise, because an admin action that
silently did nothing would be worse than an error message.
"""

from typing import Any, ClassVar, Optional

from supabase import Client


class TeamAliasService:
    """Typed access to the alias table, with the Supabase client injected."""

    TABLE: ClassVar[str] = "team_aliases"
    STATUS_PENDING: ClassVar[str] = "pending"
    STATUS_APPROVED: ClassVar[str] = "approved"
    #: Natural key — one decision per scraped name per competition.
    CONFLICT_KEY: ClassVar[str] = "league_code,raw_name"

    def __init__(self, supabase: Client) -> None:
        self._db = supabase

    def list_aliases(self, status: Optional[str] = None) -> list[dict]:
        """Every alias row, optionally narrowed to one status."""
        try:
            query = self._db.table(self.TABLE).select("*")
            if status is not None:
                query = query.eq("status", status)
            response = query.order("raw_name", desc=False).execute()
        except Exception:
            return []
        data: list[dict] = response.data or []  # type: ignore[assignment]
        return data

    def list_approved(self) -> list[dict]:
        """Only the rows an admin has confirmed."""
        return self.list_aliases(status=self.STATUS_APPROVED)

    def list_pending(self) -> list[dict]:
        """Only the rows awaiting an admin decision."""
        return self.list_aliases(status=self.STATUS_PENDING)

    def approve(
        self,
        league_code: str,
        raw_name: str,
        canonical_name: str,
        approved_by: Optional[str] = None,
    ) -> None:
        """Confirm a mapping. Upserted, so re-approving is harmless."""
        payload: dict[str, Any] = {
            "league_code": league_code,
            "raw_name": raw_name,
            "canonical_name": canonical_name,
            "status": self.STATUS_APPROVED,
            "approved_by": approved_by,
        }
        self._db.table(self.TABLE).upsert(
            payload, on_conflict=self.CONFLICT_KEY
        ).execute()

    def record_pending(self, league_code: str, raw_name: str) -> None:
        """Queue an unresolved name for review.

        Best-effort: the pipeline meets the same unknown name on every run and
        must never fail because the queue is unavailable. ``ignore_duplicates``
        keeps a re-encounter from reverting a mapping an admin already approved.
        """
        payload: dict[str, Any] = {
            "league_code": league_code,
            "raw_name": raw_name,
            "canonical_name": None,
            "status": self.STATUS_PENDING,
        }
        try:
            self._db.table(self.TABLE).upsert(
                payload, on_conflict=self.CONFLICT_KEY, ignore_duplicates=True
            ).execute()
        except Exception:
            return None

    def revoke(self, league_code: str, raw_name: str) -> None:
        """Remove a mapping entirely, returning the name to unresolved."""
        self._db.table(self.TABLE).delete().eq("league_code", league_code).eq(
            "raw_name", raw_name
        ).execute()
