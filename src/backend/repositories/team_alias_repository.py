"""Supabase-backed alias source and review queue.

Adapts :class:`src.backend.services.team_alias_service.TeamAliasService` onto
the two interfaces :mod:`src.teams.resolver` defines, so the resolver stays
free of any storage knowledge and can be driven from a YAML file, a database,
or a test double interchangeably.

Both directions are deliberately forgiving: a malformed row is dropped rather
than resolving a fixture to nothing, and a failed queue write never blocks the
pipeline that was merely reporting a name it did not recognise.
"""

from typing import Any

from src.teams.resolver import (
    Resolution,
    TeamAlias,
    TeamAliasRepository,
    UnresolvedNameSink,
)


class SupabaseTeamAliasRepository(TeamAliasRepository):
    """Admin-approved aliases stored in Supabase."""

    def __init__(self, alias_service: Any) -> None:
        self._alias_service = alias_service

    def get_aliases(self) -> list[TeamAlias]:
        try:
            rows = self._alias_service.list_approved()
        except Exception:
            return []
        return [alias for alias in map(self._to_alias, rows) if alias is not None]

    @staticmethod
    def _to_alias(row: dict) -> TeamAlias | None:
        """Drop rows that cannot resolve a name to a real team."""
        league_code = str(row.get("league_code") or "").strip()
        raw_name = str(row.get("raw_name") or "").strip()
        canonical = row.get("canonical_name")
        canonical_name = str(canonical or "").strip()
        if not league_code or not raw_name or not canonical_name:
            return None
        return TeamAlias(
            league_code=league_code,
            raw_name=raw_name,
            canonical_name=canonical_name,
        )


class SupabaseUnresolvedNameSink(UnresolvedNameSink):
    """Queues unrecognised scraped names for an admin to review."""

    def __init__(self, alias_service: Any) -> None:
        self._alias_service = alias_service

    def record(self, resolution: Resolution) -> None:
        try:
            # The *scope*, not the competition: an alias approved for a club
            # must serve every competition it plays in, and the scope is what
            # carries the country the review screen needs.
            self._alias_service.record_pending(
                league_code=resolution.scope, raw_name=resolution.raw_name
            )
        except Exception:
            return None
