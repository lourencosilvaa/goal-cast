import pytest

from src.scrapers.flashscore.http_client import FlashScoreHttpClient
from src.scrapers.flashscore.exceptions import FlashScoreHttpError
from unittest.mock import MagicMock


def _make_config(
    base_url: str = "https://www.flashscore.com",
    user_agent: str = "Mozilla/5.0",
    leagues: dict | None = None,
):
    config = MagicMock()
    config.base_url = base_url
    config.user_agent = user_agent
    config.leagues = leagues or {"E0": "england/premier-league"}
    return config


class TestFlashScoreHttpClient:

    def test_init_stores_config(self):
        config = _make_config()
        client = FlashScoreHttpClient(config)
        assert client.config is config

    def test_fetch_fixtures_raises_http_error(self):
        """HTTP client is no longer functional; always raises FlashScoreHttpError."""
        client = FlashScoreHttpClient(_make_config())
        with pytest.raises(FlashScoreHttpError):
            client.fetch_fixtures("E0")

    def test_fetch_results_raises_http_error(self):
        """HTTP client is no longer functional; always raises FlashScoreHttpError."""
        client = FlashScoreHttpClient(_make_config())
        with pytest.raises(FlashScoreHttpError):
            client.fetch_results("E0")

    def test_fetch_fixtures_raises_for_unknown_league(self):
        client = FlashScoreHttpClient(_make_config(leagues={"E0": "england/premier-league"}))
        with pytest.raises(FlashScoreHttpError, match="Unknown league"):
            client.fetch_fixtures("XX")

    def test_fetch_results_raises_for_unknown_league(self):
        client = FlashScoreHttpClient(_make_config(leagues={"E0": "england/premier-league"}))
        with pytest.raises(FlashScoreHttpError, match="Unknown league"):
            client.fetch_results("XX")

    def test_headers_include_user_agent(self):
        config = _make_config(user_agent="TestAgent/1.0")
        client = FlashScoreHttpClient(config)
        headers = client._headers()
        assert headers["User-Agent"] == "TestAgent/1.0"
        assert headers["Referer"] == config.base_url
