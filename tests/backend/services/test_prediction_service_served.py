"""Predictions are scoped to the served leagues at read time.

Filtering the catalogue endpoints is not enough. The dashboard renders the
``/api/predictions`` payload directly, and the ``predictions`` table still
holds rows for every division uploaded before a league was withdrawn —
``upsert`` never deletes. So an unserved league keeps arriving in the batch
fetch until it is filtered *here*.

Read-time filtering rather than a one-off cleanup, deliberately: it withholds
historical rows without touching them, so re-serving a division makes its
existing predictions reappear instead of having to be recomputed.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from config.config_loader import DataConfig
from src.backend.services import prediction_service as module
from src.backend.services.prediction_service import PredictionService

CORPUS = {
    "E0": "Premier League",
    "E1": "Championship",
    "SP1": "La Liga",
    "SP2": "La Liga 2",
}


def _payload(league_code: str) -> dict[str, Any]:
    return {
        "league_code": league_code,
        "league_name": CORPUS[league_code],
        "matches": [{"home_team": "A", "away_team": "B"}],
    }


class _Config:
    """Explicit config (§7.3) — corpus of four, only two of them served."""

    def __init__(self, served: list[str]) -> None:
        self.data = DataConfig(
            base_url="https://example.test/",
            leagues=dict(CORPUS),
            served_leagues=list(served),
            seasons=["2526"],
            columns_to_keep=["Date"],
        )


@pytest.fixture
def supabase(monkeypatch):
    """Stands in for the Supabase client, returning rows for all four leagues."""
    client = MagicMock()

    def _rows(*_args, **_kwargs):
        response = MagicMock()
        response.data = [
            {"league_code": code, "payload": _payload(code), "match_date": "09/08/2026"}
            for code in CORPUS
        ]
        return response

    table = client.table.return_value
    table.select.return_value = table
    table.eq.return_value = table
    table.execute.side_effect = _rows
    monkeypatch.setattr(module, "get_supabase_client", lambda: client)
    return client


class TestBatchFetchIsScopedToServedLeagues:

    def test_unserved_leagues_are_withheld(self, supabase):
        service = PredictionService(_Config(["E0", "SP1"]))
        codes = {r.league_code for r in service.get_all_leagues_predictions("09/08/2026")}
        assert codes == {"E0", "SP1"}

    def test_served_leagues_survive(self, supabase):
        service = PredictionService(_Config(["E0", "SP1"]))
        codes = {r.league_code for r in service.get_all_leagues_predictions("09/08/2026")}
        assert "E0" in codes and "SP1" in codes

    def test_serving_everything_returns_everything(self, supabase):
        service = PredictionService(_Config(list(CORPUS)))
        codes = {r.league_code for r in service.get_all_leagues_predictions("09/08/2026")}
        assert codes == set(CORPUS)

    def test_withheld_rows_do_not_poison_the_cache(self, supabase):
        """An unserved league must not be cached and then served on a hit."""
        service = PredictionService(_Config(["E0"]))
        service.get_all_leagues_predictions("09/08/2026")
        assert not any(key.startswith("E1:") for key in service._cache)


class TestSingleLeagueFetchIsScopedToo:

    def test_unserved_league_returns_no_matches(self, supabase):
        service = PredictionService(_Config(["E0"]))
        assert service.get_league_predictions("E1", "09/08/2026").matches == []

    def test_served_league_returns_its_matches(self, supabase):
        service = PredictionService(_Config(["E0"]))
        assert service.get_league_predictions("E0", "09/08/2026").matches != []

    def test_unserved_league_is_not_queried_at_all(self, supabase):
        """Withheld before the network call, not after — no wasted round trip."""
        service = PredictionService(_Config(["E0"]))
        supabase.table.reset_mock()
        service.get_league_predictions("SP2", "09/08/2026")
        supabase.table.assert_not_called()


class TestAvailableDatesAreScopedToo:

    def test_a_date_with_only_unserved_fixtures_is_not_offered(self, monkeypatch):
        """Otherwise the picker offers a day that renders as empty."""
        client = MagicMock()
        response = MagicMock()
        response.data = [
            {"match_date": "08/08/2026", "league_code": "E1"},
            {"match_date": "09/08/2026", "league_code": "E0"},
        ]
        table = client.table.return_value
        table.select.return_value = table
        table.execute.return_value = response
        monkeypatch.setattr(module, "get_supabase_client", lambda: client)

        service = PredictionService(_Config(["E0"]))
        assert service.get_available_dates() == ["09/08/2026"]

    def test_a_date_with_any_served_fixture_is_offered(self, monkeypatch):
        client = MagicMock()
        response = MagicMock()
        response.data = [
            {"match_date": "09/08/2026", "league_code": "E1"},
            {"match_date": "09/08/2026", "league_code": "E0"},
        ]
        table = client.table.return_value
        table.select.return_value = table
        table.execute.return_value = response
        monkeypatch.setattr(module, "get_supabase_client", lambda: client)

        service = PredictionService(_Config(["E0"]))
        assert service.get_available_dates() == ["09/08/2026"]
