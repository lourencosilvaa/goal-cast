"""FlashScoreScraper — one strategy, honestly named.

It used to orchestrate three collaborators: an HTTP client, a parser and a
Playwright client. Two of them did nothing. The HTTP client's own docstring
said every call raises (FlashScore moved the ``feed_sign`` token into a runtime
JS object years ago), and the exception was caught and discarded one line
later. The parser was constructed, injected, stored as ``self._parser`` — and
never called by anything.

So the scraper's real behaviour was always "render the page and read the DOM",
with two collaborators making it look like a fallback chain. Both are gone, and
these tests describe what is left.
"""

from unittest.mock import MagicMock

import pytest

from src.scrapers.base_scraper import FlashScoreFixture
from src.scrapers.flashscore.exceptions import (
    FlashScoreHttpError,
    FlashScoreUnavailableError,
)
from src.scrapers.flashscore.scraper import FlashScoreScraper


def _make_fixture(
    match_id: str = "id1", home: str = "Arsenal", away: str = "Chelsea"
) -> FlashScoreFixture:
    return FlashScoreFixture(
        match_id=match_id,
        home_team=home,
        away_team=away,
        league="Premier League",
        country="England",
        match_datetime="2026-04-29T15:00:00",
        status="scheduled",
        home_score=None,
        away_score=None,
        source_url=f"https://www.flashscore.com/match/{match_id}/",
    )


def _make_config(playwright_fallback_enabled: bool = True):
    config = MagicMock()
    config.playwright_fallback_enabled = playwright_fallback_enabled
    config.leagues = {"E0": "england/premier-league"}
    return config


def _scraper(config=None, playwright=None) -> FlashScoreScraper:
    return FlashScoreScraper(config or _make_config(), playwright or MagicMock())


class TestIdentity:
    def test_source_name(self):
        assert _scraper().source_name == "FlashScore"

    def test_available_leagues_come_from_configuration(self):
        assert _scraper().get_available_leagues() == {"E0": "england/premier-league"}

    def test_the_league_map_is_a_copy(self):
        """A caller mutating the returned map must not edit the config."""
        config = _make_config()
        _scraper(config).get_available_leagues()["E0"] = "elsewhere"
        assert config.leagues["E0"] == "england/premier-league"


class TestScrapingFixtures:
    def test_fixtures_come_from_the_playwright_client(self):
        playwright = MagicMock()
        playwright.scrape_fixtures.return_value = [_make_fixture()]
        assert _scraper(playwright=playwright).scrape_fixtures("E0") == [
            _make_fixture()
        ]

    def test_the_league_code_reaches_the_client(self):
        playwright = MagicMock()
        playwright.scrape_fixtures.return_value = []
        _scraper(playwright=playwright).scrape_fixtures("E0")
        playwright.scrape_fixtures.assert_called_once_with("E0")

    def test_an_empty_page_is_an_empty_list_not_a_failure(self):
        """A league with nothing scheduled is a normal answer."""
        playwright = MagicMock()
        playwright.scrape_fixtures.return_value = []
        assert _scraper(playwright=playwright).scrape_fixtures("E0") == []


class TestScrapingResults:
    def test_results_come_from_the_playwright_client(self):
        playwright = MagicMock()
        playwright.scrape_results.return_value = [_make_fixture()]
        assert _scraper(playwright=playwright).scrape_results("E0") == [_make_fixture()]

    def test_the_results_page_is_requested_not_the_fixtures_one(self):
        playwright = MagicMock()
        playwright.scrape_results.return_value = []
        _scraper(playwright=playwright).scrape_results("E0")
        playwright.scrape_results.assert_called_once_with("E0")
        playwright.scrape_fixtures.assert_not_called()


class TestFailures:
    def test_a_scrape_failure_is_reported_as_unavailable(self):
        playwright = MagicMock()
        playwright.scrape_fixtures.side_effect = FlashScoreHttpError("no browser")
        with pytest.raises(FlashScoreUnavailableError):
            _scraper(playwright=playwright).scrape_fixtures("E0")

    def test_the_failure_names_the_league(self):
        playwright = MagicMock()
        playwright.scrape_results.side_effect = FlashScoreHttpError("timeout")
        with pytest.raises(FlashScoreUnavailableError, match="E0"):
            _scraper(playwright=playwright).scrape_results("E0")

    def test_an_unexpected_error_is_not_swallowed(self):
        """Only FlashScore's own failure mode is translated; anything else is
        a bug here and must surface as itself."""
        playwright = MagicMock()
        playwright.scrape_fixtures.side_effect = MemoryError("out of memory")
        with pytest.raises(MemoryError):
            _scraper(playwright=playwright).scrape_fixtures("E0")

    def test_a_disabled_client_leaves_no_strategy(self):
        config = _make_config(playwright_fallback_enabled=False)
        playwright = MagicMock()
        with pytest.raises(FlashScoreUnavailableError):
            _scraper(config, playwright).scrape_fixtures("E0")
        playwright.scrape_fixtures.assert_not_called()


class TestShippedConfiguration:
    """The config the scraper is actually built from.

    Kept here rather than in a separate edge-case file: the league map is what
    decides which competitions can be scraped at all, and a missing code is a
    silently unavailable competition rather than an error.
    """

    @staticmethod
    def _flashscore():
        from config.config_loader import load_config

        return load_config("config/config.yaml").scrapers.flashscore

    def test_the_base_url_is_configured(self):
        assert self._flashscore().base_url == "https://www.flashscore.com"

    def test_the_browser_strategy_is_enabled(self):
        assert self._flashscore().playwright_fallback_enabled is True

    def test_the_dead_http_tier_is_gone_from_configuration(self):
        """A flag for a path that cannot run suggests a choice that does not
        exist."""
        assert not hasattr(self._flashscore(), "http_enabled")

    def test_league_slugs_cover_the_main_competitions(self):
        leagues = self._flashscore().leagues
        expected = {"E0", "SP1", "D1", "I1", "F1", "P1", "CL", "EL"}
        assert expected.issubset(set(leagues))
