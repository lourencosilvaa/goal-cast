"""The leagues endpoint must offer the European competitions too.

They cannot simply be added to ``data.leagues``: that mapping drives the data
loader, which downloads a football-data.co.uk CSV per code. There is no such
feed for the Champions League — that absence is the whole reason the
openfootball corpus exists — so a ``CL`` entry there would make the loader
request a file that does not exist on every run.

So the endpoint composes two sources instead, and marks which is which, since
a European competition has no league table and its fixtures come from a
different pipeline.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.config_loader import Config, DataConfig, EuropeanConfig
from src.backend.api.leagues import router
from src.backend.core.auth import get_approved_user


def _client(european: EuropeanConfig | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_approved_user] = lambda: "test-user"

    class _Config:
        data = DataConfig(
            base_url="https://example.test/",
            leagues={"E0": "Premier League", "P1": "Liga Portugal"},
            seasons=["2526"],
            columns_to_keep=["Date"],
        )

    config = _Config()
    config.european = european if european is not None else EuropeanConfig(
        enabled=True,
        competitions={"CL": "cl", "EL": "el", "UECL": "conf"},
        competition_names={
            "CL": "Champions League",
            "EL": "Europa League",
            "UECL": "Conference League",
        },
    )
    app.state.config = config
    return TestClient(app)


def _codes(payload) -> list[str]:
    return [entry["code"] for entry in payload]


class TestDomesticLeagues:
    def test_domestic_leagues_are_still_listed(self):
        payload = _client().get("/api/leagues").json()
        assert "E0" in _codes(payload)

    def test_domestic_leagues_keep_their_names(self):
        payload = _client().get("/api/leagues").json()
        entry = next(e for e in payload if e["code"] == "E0")
        assert entry["name"] == "Premier League"

    def test_domestic_leagues_are_marked_as_leagues(self):
        payload = _client().get("/api/leagues").json()
        entry = next(e for e in payload if e["code"] == "E0")
        assert entry["type"] == "league"


class TestEuropeanCompetitions:
    def test_the_competitions_are_listed(self):
        payload = _client().get("/api/leagues").json()
        assert {"CL", "EL", "UECL"} <= set(_codes(payload))

    def test_they_carry_a_readable_name(self):
        payload = _client().get("/api/leagues").json()
        entry = next(e for e in payload if e["code"] == "CL")
        assert entry["name"] == "Champions League"

    def test_they_are_marked_as_competitions(self):
        """The UI needs to know there is no league table behind these."""
        payload = _client().get("/api/leagues").json()
        entry = next(e for e in payload if e["code"] == "CL")
        assert entry["type"] == "competition"

    def test_domestic_leagues_come_first(self):
        payload = _client().get("/api/leagues").json()
        codes = _codes(payload)
        assert codes.index("E0") < codes.index("CL")

    def test_a_code_without_a_name_falls_back_to_the_code(self):
        payload = _client(
            EuropeanConfig(enabled=True, competitions={"CLQ": "clq"})
        ).get("/api/leagues").json()
        entry = next(e for e in payload if e["code"] == "CLQ")
        assert entry["name"] == "CLQ"


class TestDisabledTrack:
    def test_nothing_european_is_offered_when_the_track_is_off(self):
        payload = _client(
            EuropeanConfig(enabled=False, competitions={"CL": "cl"})
        ).get("/api/leagues").json()
        assert "CL" not in _codes(payload)

    def test_domestic_leagues_are_unaffected(self):
        payload = _client(
            EuropeanConfig(enabled=False, competitions={"CL": "cl"})
        ).get("/api/leagues").json()
        assert "E0" in _codes(payload)
