"""Tests for the Supabase-backed team-alias store.

Mirrors the shape of the existing user/app-settings services: a thin typed
wrapper over one table, with the Supabase client injected so nothing here
touches a network.

The table holds two kinds of row, distinguished by ``status``:

* ``pending``  — a scraped name the pipeline could not resolve, awaiting review
* ``approved`` — a mapping an admin confirmed; only these ever resolve a name
"""

from unittest.mock import MagicMock

import pytest

from src.backend.services.team_alias_service import TeamAliasService

LEAGUE = "P1"
RAW = "Sporting CP"
CANONICAL = "Sp Lisbon"


class FakeTable:
    """Records the query chain and returns a canned payload."""

    def __init__(self, data=None):
        self.data = data if data is not None else []
        self.calls: list[tuple] = []
        self.payload = None

    def _record(self, name, *args):
        self.calls.append((name, *args))
        return self

    def select(self, *a):
        return self._record("select", *a)

    def eq(self, *a):
        return self._record("eq", *a)

    def order(self, *a, **kw):
        return self._record("order", *a)

    def insert(self, payload):
        self.payload = payload
        return self._record("insert")

    def upsert(self, payload, **kw):
        self.payload = payload
        return self._record("upsert")

    def delete(self):
        return self._record("delete")

    def execute(self):
        response = MagicMock()
        response.data = self.data
        return response


def client_with(table: FakeTable) -> MagicMock:
    client = MagicMock()
    client.table.return_value = table
    return client


def service(table: FakeTable) -> TeamAliasService:
    return TeamAliasService(supabase=client_with(table))


class TestListing:

    def test_list_returns_rows(self):
        table = FakeTable(
            [
                {
                    "league_code": LEAGUE,
                    "raw_name": RAW,
                    "canonical_name": CANONICAL,
                    "status": "approved",
                }
            ]
        )
        assert service(table).list_aliases() == table.data

    def test_list_uses_the_team_aliases_table(self):
        table = FakeTable()
        client = client_with(table)
        TeamAliasService(supabase=client).list_aliases()
        client.table.assert_called_with(TeamAliasService.TABLE)

    def test_list_filters_by_status_when_asked(self):
        table = FakeTable()
        service(table).list_aliases(status="pending")
        assert ("eq", "status", "pending") in table.calls

    def test_list_without_status_does_not_filter(self):
        table = FakeTable()
        service(table).list_aliases()
        assert not any(call[0] == "eq" for call in table.calls)

    def test_none_data_becomes_an_empty_list(self):
        assert service(FakeTable(None)).list_aliases() == []

    def test_approved_aliases_helper_filters_by_status(self):
        table = FakeTable()
        service(table).list_approved()
        assert ("eq", "status", TeamAliasService.STATUS_APPROVED) in table.calls

    def test_pending_aliases_helper_filters_by_status(self):
        table = FakeTable()
        service(table).list_pending()
        assert ("eq", "status", TeamAliasService.STATUS_PENDING) in table.calls


class TestApproval:

    def test_approve_writes_an_approved_row(self):
        table = FakeTable()
        service(table).approve(
            league_code=LEAGUE, raw_name=RAW, canonical_name=CANONICAL
        )
        assert table.payload["league_code"] == LEAGUE
        assert table.payload["raw_name"] == RAW
        assert table.payload["canonical_name"] == CANONICAL
        assert table.payload["status"] == TeamAliasService.STATUS_APPROVED

    def test_approve_upserts_so_re_approval_is_idempotent(self):
        table = FakeTable()
        service(table).approve(
            league_code=LEAGUE, raw_name=RAW, canonical_name=CANONICAL
        )
        assert any(call[0] == "upsert" for call in table.calls)

    def test_approve_records_the_approving_admin(self):
        table = FakeTable()
        service(table).approve(
            league_code=LEAGUE,
            raw_name=RAW,
            canonical_name=CANONICAL,
            approved_by="admin-1",
        )
        assert table.payload["approved_by"] == "admin-1"


class TestPendingQueue:

    def test_record_pending_writes_a_pending_row(self):
        table = FakeTable()
        service(table).record_pending(league_code=LEAGUE, raw_name=RAW)
        assert table.payload["status"] == TeamAliasService.STATUS_PENDING
        assert table.payload["canonical_name"] is None

    def test_record_pending_is_idempotent(self):
        """The pipeline meets the same unknown name on every run."""
        table = FakeTable()
        service(table).record_pending(league_code=LEAGUE, raw_name=RAW)
        assert any(call[0] == "upsert" for call in table.calls)

    def test_record_pending_never_overwrites_an_approved_row(self):
        """Ignoring duplicates keeps an approved mapping from being reverted."""
        table = FakeTable()
        service(table).record_pending(league_code=LEAGUE, raw_name=RAW)
        upserts = [call for call in table.calls if call[0] == "upsert"]
        assert upserts, "expected an upsert"


class TestRevocation:

    def test_revoke_deletes_the_row(self):
        table = FakeTable()
        service(table).revoke(league_code=LEAGUE, raw_name=RAW)
        assert any(call[0] == "delete" for call in table.calls)

    def test_revoke_scopes_to_league_and_name(self):
        table = FakeTable()
        service(table).revoke(league_code=LEAGUE, raw_name=RAW)
        assert ("eq", "league_code", LEAGUE) in table.calls
        assert ("eq", "raw_name", RAW) in table.calls


class TestFailureHandling:

    def test_listing_failure_yields_an_empty_list(self):
        """A statistics page must not 500 because the alias table is down."""
        client = MagicMock()
        client.table.side_effect = RuntimeError("supabase down")
        assert TeamAliasService(supabase=client).list_aliases() == []

    def test_record_pending_failure_is_swallowed(self):
        client = MagicMock()
        client.table.side_effect = RuntimeError("supabase down")
        TeamAliasService(supabase=client).record_pending(
            league_code=LEAGUE, raw_name=RAW
        )  # must not raise

    def test_approve_failure_propagates(self):
        """An admin action that silently failed would be worse than an error."""
        client = MagicMock()
        client.table.side_effect = RuntimeError("supabase down")
        with pytest.raises(RuntimeError):
            TeamAliasService(supabase=client).approve(
                league_code=LEAGUE, raw_name=RAW, canonical_name=CANONICAL
            )
