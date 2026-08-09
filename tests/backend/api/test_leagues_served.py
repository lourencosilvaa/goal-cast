"""The leagues endpoint offers the served subset, not the whole corpus.

``data.leagues`` is the training corpus — every division the loader downloads
and the ELO walk spans. It is deliberately wider than the product: the
secondary divisions are half the training rows and the reason a promoted side
arrives with a real rating, so they are trained on but never offered.

The filter applies to domestic entries only. The UEFA competitions come from a
different pipeline and are not in ``data.leagues`` at all, so they cannot be
filtered by a subset of it — they pass through untouched.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.config_loader import DataConfig, EuropeanConfig
from src.backend.api.leagues import router
from src.backend.core.auth import get_approved_user

CORPUS = {
    "E0": "Premier League",
    "E1": "Championship",
    "SP1": "La Liga",
    "SP2": "La Liga 2",
}


def _client(served: list[str], *, european: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_approved_user] = lambda: "test-user"

    class _Config:
        data = DataConfig(
            base_url="https://example.test/",
            leagues=dict(CORPUS),
            served_leagues=list(served),
            seasons=["2526"],
            columns_to_keep=["Date"],
        )

    config = _Config()
    config.european = EuropeanConfig(
        enabled=european,
        competitions={"CL": "cl"},
        competition_names={"CL": "Champions League"},
    )
    app.state.config = config
    return TestClient(app, raise_server_exceptions=False)


def _codes(response) -> set[str]:
    return {entry["code"] for entry in response.json()}


class TestLeaguesAreFilteredToTheServedSet:

    def test_served_league_is_offered(self):
        assert "E0" in _codes(_client(["E0", "SP1"]).get("/api/leagues"))

    def test_unserved_league_is_withheld(self):
        codes = _codes(_client(["E0", "SP1"]).get("/api/leagues"))
        assert "E1" not in codes
        assert "SP2" not in codes

    def test_offers_exactly_the_served_domestic_set(self):
        codes = _codes(_client(["E0", "SP1"]).get("/api/leagues"))
        assert {c for c in codes if c in CORPUS} == {"E0", "SP1"}

    def test_names_survive_filtering(self):
        entries = _client(["E0"]).get("/api/leagues").json()
        assert {"code": "E0", "name": "Premier League", "type": "league"} in entries


class TestEuropeanEntriesAreUnaffected:

    def test_competition_is_offered_even_though_it_is_not_in_the_corpus(self):
        assert "CL" in _codes(_client(["E0"]).get("/api/leagues"))

    def test_competition_survives_a_single_league_served_set(self):
        codes = _codes(_client(["E0"]).get("/api/leagues"))
        assert codes == {"E0", "CL"}

    def test_disabled_european_track_still_serves_domestic(self):
        codes = _codes(_client(["E0"], european=False).get("/api/leagues"))
        assert "E0" in codes
