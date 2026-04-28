from unittest.mock import MagicMock, patch

import pytest

from src.scrapers.flashscore.playwright_client import FlashScorePlaywrightClient
from src.scrapers.flashscore.exceptions import FlashScoreHttpError


def _make_config(
    base_url: str = "https://www.flashscore.com",
    request_timeout: int = 10,
    user_agent: str = "Mozilla/5.0",
    playwright_fallback_enabled: bool = True,
    leagues: dict | None = None,
):
    config = MagicMock()
    config.base_url = base_url
    config.request_timeout = request_timeout
    config.user_agent = user_agent
    config.playwright_fallback_enabled = playwright_fallback_enabled
    config.leagues = leagues or {"E0": "england/premier-league"}
    return config


class TestFlashScorePlaywrightClient:

    def test_init_stores_config(self):
        config = _make_config()
        client = FlashScorePlaywrightClient(config)
        assert client.config is config

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_fetch_fixtures_returns_html(self, mock_playwright):
        mock_page = MagicMock()
        mock_page.content.return_value = "<html>fixture data</html>"
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_p = MagicMock()
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__ = MagicMock(return_value=mock_p)
        mock_playwright.return_value.__exit__ = MagicMock(return_value=False)

        config = _make_config()
        client = FlashScorePlaywrightClient(config)
        html = client.fetch_fixtures("E0")

        assert isinstance(html, str)
        assert "fixture data" in html

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_fetch_fixtures_raises_for_unknown_league(self, mock_playwright):
        config = _make_config(leagues={"E0": "england/premier-league"})
        client = FlashScorePlaywrightClient(config)

        with pytest.raises(FlashScoreHttpError):
            client.fetch_fixtures("XX")

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_fetch_fixtures_raises_on_playwright_error(self, mock_playwright):
        mock_playwright.return_value.__enter__ = MagicMock(
            side_effect=Exception("Playwright crashed")
        )
        mock_playwright.return_value.__exit__ = MagicMock(return_value=False)

        config = _make_config()
        client = FlashScorePlaywrightClient(config)

        with pytest.raises(FlashScoreHttpError):
            client.fetch_fixtures("E0")

    @patch("src.scrapers.flashscore.playwright_client.sync_playwright")
    def test_browser_is_closed_after_fetch(self, mock_playwright):
        mock_page = MagicMock()
        mock_page.content.return_value = "<html>data</html>"
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_p = MagicMock()
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__ = MagicMock(return_value=mock_p)
        mock_playwright.return_value.__exit__ = MagicMock(return_value=False)

        config = _make_config()
        client = FlashScorePlaywrightClient(config)
        client.fetch_fixtures("E0")

        mock_browser.close.assert_called_once()
