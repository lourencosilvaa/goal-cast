from unittest.mock import MagicMock

import pytest

from src.scrapers.flashscore.scraper import FlashScoreScraper
from src.scrapers.flashscore.exceptions import FlashScoreHttpError, FlashScoreUnavailableError
from src.scrapers.base_scraper import FlashScoreFixture


def _make_fixture(match_id: str = "id1", home: str = "Arsenal", away: str = "Chelsea") -> FlashScoreFixture:
    return FlashScoreFixture(
        match_id=match_id,
        home_team=home,
        away_team=away,
        league="Premier League",
        country="England",
        match_datetime="2024-01-01T15:00:00",
        status="scheduled",
        home_score=None,
        away_score=None,
        source_url="https://www.flashscore.com/match/id1",
    )


def _make_config(
    http_enabled: bool = True,
    playwright_fallback_enabled: bool = True,
):
    config = MagicMock()
    config.http_enabled = http_enabled
    config.playwright_fallback_enabled = playwright_fallback_enabled
    config.leagues = {"E0": "england/premier-league"}
    return config


class TestFlashScoreScraper:

    def test_source_name(self):
        config = _make_config()
        http_client = MagicMock()
        playwright_client = MagicMock()
        parser = MagicMock()
        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)
        assert scraper.source_name == "FlashScore"

    def test_scrape_fixtures_uses_http_client_first(self):
        config = _make_config()
        http_client = MagicMock()
        http_client.fetch_fixtures.return_value = "raw_data"
        playwright_client = MagicMock()
        parser = MagicMock()
        parser.parse_fixtures.return_value = [_make_fixture()]

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)
        result = scraper.scrape_fixtures("E0")

        http_client.fetch_fixtures.assert_called_once_with("E0")
        playwright_client.fetch_fixtures.assert_not_called()
        assert len(result) == 1

    def test_scrape_fixtures_falls_back_to_playwright_on_http_error(self):
        config = _make_config()
        http_client = MagicMock()
        http_client.fetch_fixtures.side_effect = FlashScoreHttpError("token expired")
        playwright_client = MagicMock()
        playwright_client.fetch_fixtures.return_value = "<html>data</html>"
        parser = MagicMock()
        parser.parse_fixtures.return_value = [_make_fixture()]

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)
        result = scraper.scrape_fixtures("E0")

        playwright_client.fetch_fixtures.assert_called_once_with("E0")
        assert len(result) == 1

    def test_scrape_fixtures_raises_unavailable_when_both_fail(self):
        config = _make_config()
        http_client = MagicMock()
        http_client.fetch_fixtures.side_effect = FlashScoreHttpError("http fail")
        playwright_client = MagicMock()
        playwright_client.fetch_fixtures.side_effect = FlashScoreHttpError("playwright fail")
        parser = MagicMock()

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)

        with pytest.raises(FlashScoreUnavailableError):
            scraper.scrape_fixtures("E0")

    def test_scrape_fixtures_skips_http_when_disabled(self):
        config = _make_config(http_enabled=False)
        http_client = MagicMock()
        playwright_client = MagicMock()
        playwright_client.fetch_fixtures.return_value = "<html>data</html>"
        parser = MagicMock()
        parser.parse_fixtures.return_value = [_make_fixture()]

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)
        scraper.scrape_fixtures("E0")

        http_client.fetch_fixtures.assert_not_called()
        playwright_client.fetch_fixtures.assert_called_once()

    def test_scrape_fixtures_raises_when_playwright_disabled_and_http_fails(self):
        config = _make_config(playwright_fallback_enabled=False)
        http_client = MagicMock()
        http_client.fetch_fixtures.side_effect = FlashScoreHttpError("fail")
        playwright_client = MagicMock()
        parser = MagicMock()

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)

        with pytest.raises(FlashScoreUnavailableError):
            scraper.scrape_fixtures("E0")

    def test_scrape_results_uses_http_client_first(self):
        config = _make_config()
        http_client = MagicMock()
        http_client.fetch_results.return_value = "raw_results"
        playwright_client = MagicMock()
        parser = MagicMock()
        parser.parse_fixtures.return_value = [_make_fixture()]

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)
        result = scraper.scrape_results("E0")

        http_client.fetch_results.assert_called_once_with("E0")
        assert len(result) == 1

    def test_scrape_results_falls_back_to_playwright(self):
        config = _make_config()
        http_client = MagicMock()
        http_client.fetch_results.side_effect = FlashScoreHttpError("fail")
        playwright_client = MagicMock()
        playwright_client.fetch_results.return_value = "<html>results</html>"
        parser = MagicMock()
        parser.parse_fixtures.return_value = [_make_fixture()]

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)
        result = scraper.scrape_results("E0")

        playwright_client.fetch_results.assert_called_once_with("E0")
        assert len(result) == 1

    def test_get_available_leagues_returns_config_leagues(self):
        config = _make_config()
        http_client = MagicMock()
        playwright_client = MagicMock()
        parser = MagicMock()

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)
        leagues = scraper.get_available_leagues()

        assert "E0" in leagues
        assert leagues["E0"] == "england/premier-league"

    def test_scrape_fixtures_returns_flashscore_fixtures(self):
        config = _make_config()
        http_client = MagicMock()
        http_client.fetch_fixtures.return_value = "raw"
        playwright_client = MagicMock()
        parser = MagicMock()
        parser.parse_fixtures.return_value = [
            _make_fixture("id1", "Arsenal", "Chelsea"),
            _make_fixture("id2", "Liverpool", "Man City"),
        ]

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)
        result = scraper.scrape_fixtures("E0")

        assert all(isinstance(f, FlashScoreFixture) for f in result)
        assert len(result) == 2
