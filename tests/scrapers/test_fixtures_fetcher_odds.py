"""Tests for optional-odds handling on Fixture and dead-code removal.

These cover the refactor to a single Bet365 source where odds may be
absent (rendered as N/A downstream) instead of a silent ``0.0``.
"""

import importlib

import pytest

from src.scrapers.fixtures_fetcher import Fixture, _parse_odd


def _fixture_with_odds() -> Fixture:
    return Fixture(
        division="E0",
        league="Premier League",
        date="01/01/2024",
        time="15:00",
        home_team="Arsenal",
        away_team="Chelsea",
        b365_home=2.0,
        b365_draw=3.5,
        b365_away=4.0,
    )


def _fixture_without_odds() -> Fixture:
    return Fixture(
        division="E0",
        league="Premier League",
        date="01/01/2024",
        time="15:00",
        home_team="Arsenal",
        away_team="Chelsea",
        b365_home=None,
        b365_draw=None,
        b365_away=None,
    )


class TestFixtureOptionalOdds:
    def test_has_odds_true_when_all_present(self) -> None:
        assert _fixture_with_odds().has_odds is True

    def test_has_odds_false_when_missing(self) -> None:
        assert _fixture_without_odds().has_odds is False

    def test_has_odds_false_when_zero(self) -> None:
        fx = _fixture_with_odds()
        fx.b365_draw = 0.0
        assert fx.has_odds is False

    def test_implied_probabilities_present_sum_to_one(self) -> None:
        probs = _fixture_with_odds().implied_probabilities()
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_implied_probabilities_empty_when_no_odds(self) -> None:
        assert _fixture_without_odds().implied_probabilities() == {}

    def test_to_dict_reports_none_odds_when_absent(self) -> None:
        d = _fixture_without_odds().to_dict()
        assert d["has_odds"] is False
        assert d["b365_odds"] == {"home": None, "draw": None, "away": None}
        assert d["implied_probabilities"] == {}


class TestParseOdd:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2.5", 2.5),
            (1.8, 1.8),
            (None, None),
            ("", None),
            ("0", None),
            (0, None),
            ("-1.5", None),
            ("not-a-number", None),
        ],
    )
    def test_parse_odd(self, raw: object, expected: float | None) -> None:
        assert _parse_odd(raw) == expected


class TestDeadCodeRemoval:
    def test_aggregatedodds_still_available(self) -> None:
        module = importlib.import_module("src.scrapers.odds_aggregator")
        assert hasattr(module, "AggregatedOdds")

    def test_oddsaggregator_class_removed(self) -> None:
        module = importlib.import_module("src.scrapers.odds_aggregator")
        assert not hasattr(module, "OddsAggregator")

    @pytest.mark.parametrize(
        "module_name",
        [
            "src.scrapers.betano_scraper",
            "src.scrapers.betclic_scraper",
            "src.scrapers.solverde_scraper",
        ],
    )
    def test_bookmaker_scrapers_removed(self, module_name: str) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)
