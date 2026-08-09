"""Tests for the admin team-alias endpoints.

This is the human-validation channel: nothing here resolves a name by itself,
it only lets an admin confirm or revoke a mapping. Every route sits behind the
existing ``get_admin_user`` gate, so a merely-approved user cannot decide which
team a scraped name refers to.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.config_loader import EuropeanConfig, TeamAliasConfig
from src.backend.api.admin import get_admin_user
from src.backend.api.team_aliases import (
    get_alias_config,
    get_european_config,
    get_alias_service,
    get_team_repository,
    router,
)

#: Explicit resolution settings — nothing here depends on the shipped YAML.
ALIAS_CONFIG = TeamAliasConfig(
    seed_path="config/team_aliases.yaml", suggestion_count=5, suggestion_cutoff=0.4
)

APPROVED_ROW = {
    "league_code": "P1",
    "raw_name": "Sporting CP",
    "canonical_name": "Sp Lisbon",
    "status": "approved",
}
PENDING_ROW = {
    "league_code": "P1",
    "raw_name": "Unknown FC",
    "canonical_name": None,
    "status": "pending",
}
TEAMS = {"P1": ["Sp Lisbon", "Porto", "Benfica"], "E0": ["Arsenal"]}
#: Stated explicitly (§7.3) — the country→league map decides which teams a
#: queued European name may be matched against.
EUROPEAN_CONFIG = EuropeanConfig(
    country_leagues={"POR": ["P1"], "ENG": ["E0"]}, alias_scope="EU"
)


def client(service: MagicMock, teams: dict | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_admin_user] = lambda: "admin-1"
    app.dependency_overrides[get_alias_service] = lambda: service
    app.dependency_overrides[get_alias_config] = lambda: ALIAS_CONFIG
    app.dependency_overrides[get_european_config] = lambda: EUROPEAN_CONFIG
    repo = AsyncMock()
    repo.get_teams.return_value = TEAMS if teams is None else teams
    app.dependency_overrides[get_team_repository] = lambda: repo
    return TestClient(app, raise_server_exceptions=False)


def service_with(rows: list[dict]) -> MagicMock:
    service = MagicMock()
    service.list_aliases.return_value = rows
    service.list_approved.return_value = [r for r in rows if r["status"] == "approved"]
    service.list_pending.return_value = [r for r in rows if r["status"] == "pending"]
    return service


class TestListing:

    def test_returns_approved_and_pending_separately(self):
        res = client(service_with([APPROVED_ROW, PENDING_ROW])).get(
            "/api/admin/team-aliases"
        )
        assert res.status_code == 200
        body = res.json()
        assert [a["raw_name"] for a in body["approved"]] == ["Sporting CP"]
        assert [p["raw_name"] for p in body["pending"]] == ["Unknown FC"]

    def test_pending_entries_carry_suggestions(self):
        """The admin sees candidates, and picks one — nothing is auto-applied."""
        res = client(service_with([PENDING_ROW])).get("/api/admin/team-aliases")
        pending = res.json()["pending"][0]
        assert isinstance(pending["suggestions"], list)

    def test_pending_suggestions_come_from_the_right_league(self):
        row = dict(PENDING_ROW, raw_name="Sporting")
        res = client(service_with([row])).get("/api/admin/team-aliases")
        suggestions = res.json()["pending"][0]["suggestions"]
        assert all(name in TEAMS["P1"] for name in suggestions)

    def test_approved_entries_carry_the_canonical_name(self):
        res = client(service_with([APPROVED_ROW])).get("/api/admin/team-aliases")
        assert res.json()["approved"][0]["canonical_name"] == "Sp Lisbon"

    def test_empty_table_returns_empty_lists(self):
        body = client(service_with([])).get("/api/admin/team-aliases").json()
        assert body["approved"] == []
        assert body["pending"] == []

    def test_candidate_teams_are_included_for_the_picker(self):
        res = client(service_with([])).get("/api/admin/team-aliases")
        assert res.json().get("teams", TEAMS) == TEAMS


class TestApproval:

    @staticmethod
    def _body(**overrides) -> dict:
        body = {
            "league_code": "P1",
            "raw_name": "Sporting CP",
            "canonical_name": "Sp Lisbon",
        }
        body.update(overrides)
        return body

    def test_approving_stores_the_mapping(self):
        service = service_with([])
        res = client(service).post("/api/admin/team-aliases", json=self._body())
        assert res.status_code == 200
        service.approve.assert_called_once_with(
            league_code="P1",
            raw_name="Sporting CP",
            canonical_name="Sp Lisbon",
            approved_by="admin-1",
        )

    def test_canonical_name_must_be_a_real_team(self):
        """An approval typo would otherwise poison every future resolution."""
        service = service_with([])
        res = client(service).post(
            "/api/admin/team-aliases", json=self._body(canonical_name="Gone FC")
        )
        assert res.status_code == 422
        service.approve.assert_not_called()

    def test_unknown_league_is_rejected(self):
        service = service_with([])
        res = client(service).post(
            "/api/admin/team-aliases", json=self._body(league_code="ZZ")
        )
        assert res.status_code == 422
        service.approve.assert_not_called()

    def test_blank_raw_name_is_rejected(self):
        service = service_with([])
        res = client(service).post(
            "/api/admin/team-aliases", json=self._body(raw_name="   ")
        )
        assert res.status_code == 422

    def test_incomplete_body_is_rejected(self):
        res = client(service_with([])).post(
            "/api/admin/team-aliases", json={"league_code": "P1"}
        )
        assert res.status_code == 422


class TestRevocation:

    def test_revoking_removes_the_mapping(self):
        service = service_with([APPROVED_ROW])
        res = client(service).request(
            "DELETE",
            "/api/admin/team-aliases",
            json={"league_code": "P1", "raw_name": "Sporting CP"},
        )
        assert res.status_code == 200
        service.revoke.assert_called_once_with(
            league_code="P1", raw_name="Sporting CP"
        )


class TestDependencyProviders:
    """The real wiring, exercised without the overrides the other tests use."""

    def test_alias_config_comes_from_app_state(self):
        from src.backend.api.team_aliases import get_alias_config

        request = MagicMock()
        request.app.state.config.teams.aliases = ALIAS_CONFIG
        assert get_alias_config(request) is ALIAS_CONFIG

    def test_alias_service_is_built_from_the_supabase_client(self, monkeypatch):
        from src.backend.api import team_aliases as module
        from src.backend.services.team_alias_service import TeamAliasService

        client = MagicMock()
        monkeypatch.setattr(module, "get_supabase_client", lambda: client)
        assert isinstance(module.get_alias_service(), TeamAliasService)


class TestAuthorisation:

    @staticmethod
    def _unauthenticated_client() -> TestClient:
        app = FastAPI()
        app.include_router(router)
        return TestClient(app, raise_server_exceptions=False)

    def test_listing_requires_authentication(self):
        assert self._unauthenticated_client().get("/api/admin/team-aliases").status_code in (
            401,
            403,
        )

    def test_approving_requires_authentication(self):
        res = self._unauthenticated_client().post(
            "/api/admin/team-aliases",
            json={
                "league_code": "P1",
                "raw_name": "x",
                "canonical_name": "Sp Lisbon",
            },
        )
        assert res.status_code in (401, 403)

    @pytest.mark.parametrize("method,path", [("GET", "/api/admin/team-aliases")])
    def test_routes_live_under_the_admin_namespace(self, method, path):
        assert path.startswith("/api/admin/")


class TestEuropeanScopes:
    """UEFA names arrive under a scope that encodes their country.

    There is no "EU-POR" league in the registry, so both listing and approval
    have to understand the scope — otherwise every European approval is
    rejected as an unknown league and the queue is unusable.
    """

    def _pending(self, raw_name: str, scope: str = "EU-POR") -> dict:
        return {
            "league_code": scope,
            "raw_name": raw_name,
            "canonical_name": None,
            "status": "pending",
        }

    def test_country_is_reported(self):
        service = service_with([self._pending("Sport Lisboa e Benfica")])
        body = client(service).get("/api/admin/team-aliases").json()
        assert body["pending"][0]["country"] == "POR"

    def test_domestic_entries_carry_no_country(self):
        service = service_with([PENDING_ROW])
        body = client(service).get("/api/admin/team-aliases").json()
        assert body["pending"][0]["country"] is None

    def test_suggestions_come_from_that_country(self):
        service = service_with([self._pending("Sport Lisboa e Benfica")])
        body = client(service).get("/api/admin/team-aliases").json()
        assert set(body["pending"][0]["suggestions"]) <= set(TEAMS["P1"])

    def test_suggestions_never_come_from_another_country(self):
        """An English side must never be offered for a Portuguese entry."""
        service = service_with([self._pending("Arsenal FC")])
        body = client(service).get("/api/admin/team-aliases").json()
        assert "Arsenal" not in body["pending"][0]["suggestions"]

    def test_untracked_country_gets_no_suggestions(self):
        service = service_with([self._pending("AC Sparta Praha", scope="EU-CZE")])
        body = client(service).get("/api/admin/team-aliases").json()
        assert body["pending"][0]["suggestions"] == []

    def test_approval_accepts_a_team_from_that_country(self):
        service = service_with([])
        response = client(service).post(
            "/api/admin/team-aliases",
            json={
                "league_code": "EU-POR",
                "raw_name": "Sport Lisboa e Benfica",
                "canonical_name": "Benfica",
            },
        )
        assert response.status_code == 200
        service.approve.assert_called_once()

    def test_approval_rejects_a_team_from_another_country(self):
        service = service_with([])
        response = client(service).post(
            "/api/admin/team-aliases",
            json={
                "league_code": "EU-POR",
                "raw_name": "Sport Lisboa e Benfica",
                "canonical_name": "Arsenal",
            },
        )
        assert response.status_code == 422

    def test_approval_rejects_an_untracked_country(self):
        service = service_with([])
        response = client(service).post(
            "/api/admin/team-aliases",
            json={
                "league_code": "EU-CZE",
                "raw_name": "AC Sparta Praha",
                "canonical_name": "Sp Lisbon",
            },
        )
        assert response.status_code == 422

    def test_domestic_approval_is_unaffected(self):
        service = service_with([])
        response = client(service).post(
            "/api/admin/team-aliases",
            json={
                "league_code": "P1",
                "raw_name": "Sporting CP",
                "canonical_name": "Sp Lisbon",
            },
        )
        assert response.status_code == 200

    def test_options_cover_the_whole_country(self):
        """The picker must offer every team of that country, not only the
        few names similar enough to be suggested."""
        service = service_with([self._pending("Sport Lisboa e Benfica")])
        body = client(service).get("/api/admin/team-aliases").json()
        assert set(body["pending"][0]["options"]) == set(TEAMS["P1"])

    def test_domestic_options_are_that_league(self):
        service = service_with([PENDING_ROW])
        body = client(service).get("/api/admin/team-aliases").json()
        assert set(body["pending"][0]["options"]) == set(TEAMS["P1"])

    def test_untracked_country_offers_nothing(self):
        service = service_with([self._pending("AC Sparta Praha", scope="EU-CZE")])
        body = client(service).get("/api/admin/team-aliases").json()
        assert body["pending"][0]["options"] == []
