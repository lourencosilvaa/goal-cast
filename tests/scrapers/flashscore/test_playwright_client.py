from unittest.mock import MagicMock, patch

import pytest

from src.scrapers.base_scraper import FlashScoreFixture
from src.scrapers.flashscore.playwright_client import FlashScorePlaywrightClient
from src.scrapers.flashscore.exceptions import FlashScoreHttpError

_RAW_ROWS = [
    {"matchId": "abc123", "home": "Sporting CP", "away": "Tondela", "time": "29.04. 20:15"},
    {"matchId": "def456", "home": "FC Porto", "away": "Benfica", "time": "30.04. 19:00"},
]


def _make_config(
    base_url: str = "https://www.flashscore.com",
    request_timeout: int = 10,
    user_agent: str = "Mozilla/5.0",
    leagues: dict | None = None,
):
    config = MagicMock()
    config.base_url = base_url
    config.request_timeout = request_timeout
    config.user_agent = user_agent
    config.leagues = leagues or {"P1": "portugal/liga-portugal", "E0": "england/premier-league"}
    return config


def _make_playwright_mock(evaluate_return: list):
    """Build a Playwright mock where page.evaluate() returns the given rows."""
    mock_page = MagicMock()
    mock_page.evaluate.return_value = evaluate_return

    mock_context = MagicMock()
    mock_context.new_page.return_value = mock_page

    mock_browser = MagicMock()
    mock_browser.new_context.return_value = mock_context

    mock_p = MagicMock()
    mock_p.chromium.launch.return_value = mock_browser

    mock_pw = MagicMock()
    mock_pw.__enter__ = MagicMock(return_value=mock_p)
    mock_pw.__exit__ = MagicMock(return_value=False)

    return mock_pw, mock_browser, mock_page


class TestFlashScorePlaywrightClientInit:

    def test_init_stores_config(self):
        config = _make_config()
        client = FlashScorePlaywrightClient(config)
        assert client.config is config


class TestFlashScorePlaywrightClientScrapeFixtures:

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_scrape_fixtures_returns_list_of_fixture_objects(self, mock_playwright):
        mock_pw, _, _ = _make_playwright_mock(_RAW_ROWS)
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())
        result = client.scrape_fixtures("P1")

        assert isinstance(result, list)
        assert all(isinstance(f, FlashScoreFixture) for f in result)
        assert len(result) == 2

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_scrape_fixtures_populates_team_names(self, mock_playwright):
        mock_pw, _, _ = _make_playwright_mock(_RAW_ROWS)
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())
        result = client.scrape_fixtures("P1")

        assert result[0].home_team == "Sporting CP"
        assert result[0].away_team == "Tondela"

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_scrape_fixtures_populates_match_id(self, mock_playwright):
        mock_pw, _, _ = _make_playwright_mock(_RAW_ROWS)
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())
        result = client.scrape_fixtures("P1")

        assert result[0].match_id == "abc123"

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_scrape_fixtures_derives_league_and_country_from_slug(self, mock_playwright):
        mock_pw, _, _ = _make_playwright_mock(_RAW_ROWS[:1])
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())
        result = client.scrape_fixtures("P1")

        assert result[0].league == "Liga Portugal"
        assert result[0].country == "Portugal"

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_scrape_fixtures_status_is_scheduled(self, mock_playwright):
        mock_pw, _, _ = _make_playwright_mock(_RAW_ROWS[:1])
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())
        result = client.scrape_fixtures("P1")

        assert result[0].status == "scheduled"

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_scrape_fixtures_parses_datetime(self, mock_playwright):
        mock_pw, _, _ = _make_playwright_mock(_RAW_ROWS[:1])
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())
        result = client.scrape_fixtures("P1")

        assert "2026-04-29" in result[0].match_datetime
        assert "20:15" in result[0].match_datetime

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_scrape_fixtures_empty_page_returns_empty_list(self, mock_playwright):
        mock_pw, _, _ = _make_playwright_mock([])
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())
        result = client.scrape_fixtures("P1")

        assert result == []

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_scrape_fixtures_raises_for_unknown_league(self, mock_playwright):
        client = FlashScorePlaywrightClient(_make_config())

        with pytest.raises(FlashScoreHttpError, match="Unknown league"):
            client.scrape_fixtures("XX")

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_scrape_fixtures_wraps_playwright_exception(self, mock_playwright):
        mock_pw = MagicMock()
        mock_pw.__enter__ = MagicMock(side_effect=Exception("browser crashed"))
        mock_pw.__exit__ = MagicMock(return_value=False)
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())

        with pytest.raises(FlashScoreHttpError):
            client.scrape_fixtures("P1")

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_browser_closed_after_scrape(self, mock_playwright):
        mock_pw, mock_browser, _ = _make_playwright_mock(_RAW_ROWS)
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())
        client.scrape_fixtures("P1")

        mock_browser.close.assert_called_once()

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_scrape_fixtures_navigates_to_fixtures_page(self, mock_playwright):
        mock_pw, _, mock_page = _make_playwright_mock(_RAW_ROWS)
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())
        client.scrape_fixtures("P1")

        call_url = mock_page.goto.call_args[0][0]
        assert "portugal/liga-portugal" in call_url
        assert "fixtures" in call_url

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_scrape_results_navigates_to_results_page(self, mock_playwright):
        mock_pw, _, mock_page = _make_playwright_mock(_RAW_ROWS)
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())
        client.scrape_results("P1")

        call_url = mock_page.goto.call_args[0][0]
        assert "results" in call_url

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_scrape_results_status_is_finished(self, mock_playwright):
        mock_pw, _, _ = _make_playwright_mock(_RAW_ROWS[:1])
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())
        result = client.scrape_results("P1")

        assert result[0].status == "finished"

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_source_url_contains_match_id(self, mock_playwright):
        mock_pw, _, _ = _make_playwright_mock(_RAW_ROWS[:1])
        mock_playwright.return_value = mock_pw

        client = FlashScorePlaywrightClient(_make_config())
        result = client.scrape_fixtures("P1")

        assert "abc123" in result[0].source_url


class TestFlashScorePlaywrightClientLabelDerivation:

    def test_derive_labels_portugal_liga_portugal(self):
        client = FlashScorePlaywrightClient(_make_config())
        league, country = client._derive_labels("portugal/liga-portugal")
        assert league == "Liga Portugal"
        assert country == "Portugal"

    def test_derive_labels_england_premier_league(self):
        client = FlashScorePlaywrightClient(_make_config())
        league, country = client._derive_labels("england/premier-league")
        assert league == "Premier League"
        assert country == "England"

    def test_derive_labels_europe_champions_league(self):
        client = FlashScorePlaywrightClient(_make_config())
        league, country = client._derive_labels("europe/champions-league")
        assert league == "Champions League"
        assert country == "Europe"

    def test_parse_time_valid(self):
        client = FlashScorePlaywrightClient(_make_config())
        result = client._parse_time("29.04. 20:15")
        assert "2026-04-29" in result
        assert "20:15" in result

    def test_parse_time_empty_returns_empty(self):
        client = FlashScorePlaywrightClient(_make_config())
        assert client._parse_time("") == ""

    def test_parse_time_invalid_returns_raw(self):
        client = FlashScorePlaywrightClient(_make_config())
        result = client._parse_time("not-a-time")
        assert result == "not-a-time"
