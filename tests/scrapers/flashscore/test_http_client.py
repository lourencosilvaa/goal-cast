from unittest.mock import MagicMock, patch

import pytest

from src.scrapers.flashscore.http_client import FlashScoreHttpClient
from src.scrapers.flashscore.exceptions import FlashScoreHttpError


def _make_config(
    base_url: str = "https://www.flashscore.com",
    api_url: str = "https://d.flashscore.com/x/feed",
    token_url: str = "https://www.flashscore.com",
    request_timeout: int = 10,
    user_agent: str = "Mozilla/5.0",
    http_enabled: bool = True,
    playwright_fallback_enabled: bool = True,
    leagues: dict | None = None,
):
    config = MagicMock()
    config.base_url = base_url
    config.api_url = api_url
    config.token_url = token_url
    config.request_timeout = request_timeout
    config.user_agent = user_agent
    config.http_enabled = http_enabled
    config.playwright_fallback_enabled = playwright_fallback_enabled
    config.leagues = leagues or {"E0": "england/premier-league"}
    return config


class TestFlashScoreHttpClient:

    def test_init_stores_config(self):
        config = _make_config()
        client = FlashScoreHttpClient(config)
        assert client.config is config

    @patch("src.scrapers.flashscore.http_client.requests.get")
    def test_get_token_extracts_from_page(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '<script>var token = "testtoken99";</script>'
        mock_get.return_value = mock_resp

        config = _make_config()
        client = FlashScoreHttpClient(config)
        token = client.get_token()

        assert token == "testtoken99"

    @patch("src.scrapers.flashscore.http_client.requests.get")
    def test_get_token_raises_on_http_error(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        config = _make_config()
        client = FlashScoreHttpClient(config)

        with pytest.raises(FlashScoreHttpError):
            client.get_token()

    @patch("src.scrapers.flashscore.http_client.requests.get")
    def test_get_token_raises_when_token_missing(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html>no token here</html>"
        mock_get.return_value = mock_resp

        config = _make_config()
        client = FlashScoreHttpClient(config)

        with pytest.raises(FlashScoreHttpError):
            client.get_token()

    @patch("src.scrapers.flashscore.http_client.requests.get")
    def test_fetch_fixtures_returns_raw_text(self, mock_get):
        token_resp = MagicMock()
        token_resp.status_code = 200
        token_resp.text = '<script>var token = "tok123";</script>'

        feed_resp = MagicMock()
        feed_resp.status_code = 200
        feed_resp.text = "SA\xf71\xac~ZEE\xf7id1\xacAA\xf7Arsenal\xacAB\xf7Chelsea\xac~"
        feed_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [token_resp, feed_resp]

        config = _make_config()
        client = FlashScoreHttpClient(config)
        raw = client.fetch_fixtures("E0")

        assert isinstance(raw, str)
        assert len(raw) > 0

    @patch("src.scrapers.flashscore.http_client.requests.get")
    def test_fetch_fixtures_raises_for_unknown_league(self, mock_get):
        token_resp = MagicMock()
        token_resp.status_code = 200
        token_resp.text = '<script>var token = "tok123";</script>'
        mock_get.return_value = token_resp

        config = _make_config(leagues={"E0": "england/premier-league"})
        client = FlashScoreHttpClient(config)

        with pytest.raises(FlashScoreHttpError):
            client.fetch_fixtures("XX")

    @patch("src.scrapers.flashscore.http_client.requests.get")
    def test_fetch_fixtures_raises_on_bad_status(self, mock_get):
        token_resp = MagicMock()
        token_resp.status_code = 200
        token_resp.text = '<script>var token = "tok123";</script>'

        feed_resp = MagicMock()
        feed_resp.raise_for_status = MagicMock(side_effect=Exception("403"))

        mock_get.side_effect = [token_resp, feed_resp]

        config = _make_config()
        client = FlashScoreHttpClient(config)

        with pytest.raises(FlashScoreHttpError):
            client.fetch_fixtures("E0")

    @patch("src.scrapers.flashscore.http_client.requests.get")
    def test_token_is_cached_across_calls(self, mock_get):
        token_resp = MagicMock()
        token_resp.status_code = 200
        token_resp.text = '<script>var token = "cached_token";</script>'

        feed_resp = MagicMock()
        feed_resp.status_code = 200
        feed_resp.text = "raw_data"
        feed_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [token_resp, feed_resp, feed_resp]

        config = _make_config()
        client = FlashScoreHttpClient(config)
        client.fetch_fixtures("E0")
        client.fetch_fixtures("E0")

        token_calls = [
            c for c in mock_get.call_args_list
            if c.args and c.args[0] == config.token_url
        ]
        assert len(token_calls) == 1
