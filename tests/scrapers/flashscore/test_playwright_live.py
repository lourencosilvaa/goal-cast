"""``scrape_live`` — the score board page, read the same way as fixtures.

Additive by design: the fixture and results pages keep the extraction script
they already had, because the results track must not be able to break the
fixture pipeline. What is new is a third page template and a script that also
reads the score cells and the stage cell.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.scrapers.flashscore.exceptions import FlashScoreHttpError
from src.scrapers.flashscore.live_rows import LiveScoreRow
from src.scrapers.flashscore.playwright_client import FlashScorePlaywrightClient

_RAW_LIVE = [
    {
        "matchId": "abc123",
        "home": "Porto",
        "away": "Alverca",
        "homeScore": "2",
        "awayScore": "0",
        "stage": "67'",
    },
    {
        "matchId": "def456",
        "home": "Benfica",
        "away": "Braga",
        "homeScore": "",
        "awayScore": "",
        "stage": "20:15",
    },
]


def _config(leagues: dict | None = None):
    config = MagicMock()
    config.base_url = "https://www.flashscore.com"
    config.request_timeout = 10
    config.user_agent = "Mozilla/5.0"
    config.leagues = leagues or {"P1": "portugal/liga-portugal"}
    return config


def _playwright(rows):
    page = MagicMock()
    page.evaluate.return_value = rows
    context = MagicMock()
    context.new_page.return_value = page
    browser = MagicMock()
    browser.new_context.return_value = context
    api = MagicMock()
    api.chromium.launch.return_value = browser
    manager = MagicMock()
    manager.__enter__ = MagicMock(return_value=api)
    manager.__exit__ = MagicMock(return_value=False)
    return manager, page


class TestScrapeLive:
    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_rows_come_back_as_live_score_rows(self, sync_playwright):
        sync_playwright.return_value = _playwright(_RAW_LIVE)[0]
        rows = FlashScorePlaywrightClient(_config()).scrape_live("P1")
        assert all(isinstance(row, LiveScoreRow) for row in rows)

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_a_scored_row_carries_integer_goals(self, sync_playwright):
        sync_playwright.return_value = _playwright(_RAW_LIVE)[0]
        row = FlashScorePlaywrightClient(_config()).scrape_live("P1")[0]
        assert (row.home_goals, row.away_goals) == (2, 0)

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_the_stage_cell_becomes_the_minute(self, sync_playwright):
        sync_playwright.return_value = _playwright(_RAW_LIVE)[0]
        row = FlashScorePlaywrightClient(_config()).scrape_live("P1")[0]
        assert row.minute == "67'"

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_an_unscored_row_keeps_none_rather_than_zero(self, sync_playwright):
        sync_playwright.return_value = _playwright(_RAW_LIVE)[0]
        row = FlashScorePlaywrightClient(_config()).scrape_live("P1")[1]
        assert (row.home_goals, row.away_goals) == (None, None)

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_rows_without_both_teams_are_dropped(self, sync_playwright):
        sync_playwright.return_value = _playwright(
            [{"matchId": "x", "home": "Porto", "away": "", "stage": "12'"}]
        )[0]
        assert FlashScorePlaywrightClient(_config()).scrape_live("P1") == []

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_the_live_page_of_the_configured_slug_is_opened(self, sync_playwright):
        manager, page = _playwright(_RAW_LIVE)
        sync_playwright.return_value = manager
        FlashScorePlaywrightClient(_config()).scrape_live("P1")
        assert "portugal/liga-portugal" in page.goto.call_args[0][0]

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_an_unknown_league_is_refused_before_a_browser_starts(
        self, sync_playwright
    ):
        with pytest.raises(FlashScoreHttpError):
            FlashScorePlaywrightClient(_config()).scrape_live("ZZ")
        sync_playwright.assert_not_called()

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_a_browser_failure_is_reported_as_a_flashscore_error(
        self, sync_playwright
    ):
        sync_playwright.side_effect = RuntimeError("no chromium")
        with pytest.raises(FlashScoreHttpError):
            FlashScorePlaywrightClient(_config()).scrape_live("P1")

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_a_non_numeric_score_is_not_guessed(self, sync_playwright):
        sync_playwright.return_value = _playwright(
            [
                {
                    "matchId": "x",
                    "home": "Porto",
                    "away": "Braga",
                    "homeScore": "-",
                    "awayScore": "-",
                    "stage": "Postponed",
                }
            ]
        )[0]
        row = FlashScorePlaywrightClient(_config()).scrape_live("P1")[0]
        assert row.home_goals is None
