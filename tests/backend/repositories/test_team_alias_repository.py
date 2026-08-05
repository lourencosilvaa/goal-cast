"""Tests for the Supabase alias repository and the unresolved-name queue.

The repository adapts :class:`TeamAliasService` rows onto the interface the
resolver expects, and is the only place approved rows become ``TeamAlias``
objects. Malformed rows are dropped rather than propagated, because a row with
a missing canonical name would resolve a fixture to ``None``.
"""

from unittest.mock import MagicMock

from src.backend.repositories.team_alias_repository import (
    SupabaseTeamAliasRepository,
    SupabaseUnresolvedNameSink,
)
from src.teams.resolver import (
    Resolution,
    TeamAlias,
    TeamAliasRepository,
    UnresolvedNameSink,
)


def row(**overrides) -> dict:
    data = {
        "league_code": "P1",
        "raw_name": "Sporting CP",
        "canonical_name": "Sp Lisbon",
        "status": "approved",
    }
    data.update(overrides)
    return data


class TestSupabaseTeamAliasRepository:

    def test_implements_the_resolver_interface(self):
        assert isinstance(SupabaseTeamAliasRepository(MagicMock()), TeamAliasRepository)

    def test_maps_rows_onto_team_aliases(self):
        service = MagicMock()
        service.list_approved.return_value = [row()]
        aliases = SupabaseTeamAliasRepository(service).get_aliases()
        assert aliases == [TeamAlias("P1", "Sporting CP", "Sp Lisbon")]

    def test_reads_only_approved_rows(self):
        service = MagicMock()
        service.list_approved.return_value = []
        SupabaseTeamAliasRepository(service).get_aliases()
        service.list_approved.assert_called_once_with()

    def test_rows_without_a_canonical_name_are_dropped(self):
        service = MagicMock()
        service.list_approved.return_value = [row(canonical_name=None), row()]
        aliases = SupabaseTeamAliasRepository(service).get_aliases()
        assert aliases == [TeamAlias("P1", "Sporting CP", "Sp Lisbon")]

    def test_rows_missing_a_league_are_dropped(self):
        service = MagicMock()
        service.list_approved.return_value = [row(league_code="")]
        assert SupabaseTeamAliasRepository(service).get_aliases() == []

    def test_service_failure_yields_no_aliases(self):
        service = MagicMock()
        service.list_approved.side_effect = RuntimeError("supabase down")
        assert SupabaseTeamAliasRepository(service).get_aliases() == []

    def test_empty_table_yields_no_aliases(self):
        service = MagicMock()
        service.list_approved.return_value = []
        assert SupabaseTeamAliasRepository(service).get_aliases() == []


class TestSupabaseUnresolvedNameSink:

    @staticmethod
    def _resolution() -> Resolution:
        return Resolution(raw_name="Unknown FC", league_code="P1")

    def test_implements_the_sink_interface(self):
        assert isinstance(SupabaseUnresolvedNameSink(MagicMock()), UnresolvedNameSink)

    def test_queues_the_name_for_review(self):
        service = MagicMock()
        SupabaseUnresolvedNameSink(service).record(self._resolution())
        service.record_pending.assert_called_once_with(
            league_code="P1", raw_name="Unknown FC"
        )

    def test_service_failure_is_swallowed(self):
        service = MagicMock()
        service.record_pending.side_effect = RuntimeError("supabase down")
        SupabaseUnresolvedNameSink(service).record(self._resolution())  # must not raise
