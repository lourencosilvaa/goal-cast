from unittest.mock import MagicMock

import pytest

from src.scrapers.flashscore.scraper import FlashScoreScraper
from src.scrapers.flashscore.exceptions import FlashScoreHttpError, FlashScoreUnavailableError
from src.scrapers.base_scraper import FlashScoreFixture


def _make_config(
    http_enabled: bool = True,
    playwright_fallback_enabled: bool = True,
    leagues: dict | None = None,
):
    config = MagicMock()
    config.http_enabled = http_enabled
    config.playwright_fallback_enabled = playwright_fallback_enabled
    config.leagues = leagues or {"E0": "england/premier-league", "SP1": "spain/laliga"}
    return config


def _make_fixture(match_id: str = "id1") -> FlashScoreFixture:
    return FlashScoreFixture(
        match_id=match_id,
        home_team="Arsenal",
        away_team="Chelsea",
        league="Premier League",
        country="England",
        match_datetime="2024-01-01T15:00:00+00:00",
        status="scheduled",
        home_score=None,
        away_score=None,
        source_url=f"https://www.flashscore.com/match/{match_id}/",
    )


class TestFlashScoreScraperEdgeCases:

    def test_scrape_fixtures_empty_parser_result_returns_empty_list(self):
        config = _make_config()
        http_client = MagicMock()
        http_client.fetch_fixtures.return_value = ""
        playwright_client = MagicMock()
        parser = MagicMock()
        parser.parse_fixtures.return_value = []

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)
        result = scraper.scrape_fixtures("E0")

        assert result == []

    def test_both_disabled_raises_unavailable(self):
        config = _make_config(http_enabled=False, playwright_fallback_enabled=False)
        http_client = MagicMock()
        playwright_client = MagicMock()
        parser = MagicMock()

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)

        with pytest.raises(FlashScoreUnavailableError):
            scraper.scrape_fixtures("E0")

    def test_scrape_results_raises_unavailable_when_both_fail(self):
        config = _make_config()
        http_client = MagicMock()
        http_client.fetch_results.side_effect = FlashScoreHttpError("fail")
        playwright_client = MagicMock()
        playwright_client.fetch_results.side_effect = FlashScoreHttpError("fail")
        parser = MagicMock()

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)

        with pytest.raises(FlashScoreUnavailableError):
            scraper.scrape_results("E0")

    def test_league_slug_drives_label_extraction(self):
        config = _make_config(leagues={"SP1": "spain/laliga"})
        http_client = MagicMock()
        http_client.fetch_fixtures.return_value = "raw"
        playwright_client = MagicMock()

        captured: dict = {}

        def capture_parse(raw, *, league, country):
            captured["league"] = league
            captured["country"] = country
            return [_make_fixture()]

        parser = MagicMock()
        parser.parse_fixtures.side_effect = capture_parse

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)
        scraper.scrape_fixtures("SP1")

        assert captured["league"] == "Laliga"
        assert captured["country"] == "Spain"

    def test_http_non_flashscore_error_propagates(self):
        config = _make_config()
        http_client = MagicMock()
        http_client.fetch_fixtures.side_effect = RuntimeError("unexpected crash")
        playwright_client = MagicMock()
        parser = MagicMock()

        scraper = FlashScoreScraper(config, http_client, playwright_client, parser)

        with pytest.raises(RuntimeError):
            scraper.scrape_fixtures("E0")

    def test_get_available_leagues_is_copy(self):
        config = _make_config()
        scraper = FlashScoreScraper(config, MagicMock(), MagicMock(), MagicMock())
        leagues = scraper.get_available_leagues()
        leagues["NEW"] = "new/league"
        assert "NEW" not in scraper.get_available_leagues()


class TestFlashScoreParserEdgeCases:

    def test_entry_without_home_team_is_skipped(self):
        from src.scrapers.flashscore.parser import FlashScoreParser
        raw = "ZEE\xf7id1\xacAB\xf7Chelsea\xacAD\xf71714000000\xacAE\xf7scheduled\xac~"
        parser = FlashScoreParser()
        result = parser.parse_fixtures(raw, league="PL", country="England")
        assert result == []

    def test_entry_without_away_team_is_skipped(self):
        from src.scrapers.flashscore.parser import FlashScoreParser
        raw = "ZEE\xf7id1\xacAA\xf7Arsenal\xacAD\xf71714000000\xacAE\xf7scheduled\xac~"
        parser = FlashScoreParser()
        result = parser.parse_fixtures(raw, league="PL", country="England")
        assert result == []

    def test_entry_without_match_id_is_skipped(self):
        from src.scrapers.flashscore.parser import FlashScoreParser
        raw = "ZEE\xf7\xacAA\xf7Arsenal\xacAB\xf7Chelsea\xacAE\xf7scheduled\xac~"
        parser = FlashScoreParser()
        result = parser.parse_fixtures(raw, league="PL", country="England")
        assert result == []

    def test_invalid_score_returns_none(self):
        from src.scrapers.flashscore.parser import FlashScoreParser
        raw = (
            "ZEE\xf7id1\xacAA\xf7Arsenal\xacAB\xf7Chelsea\xac"
            "AD\xf71714000000\xacAE\xf7finished\xacAF\xf7N/A\xacAG\xf7N/A\xac~"
        )
        parser = FlashScoreParser()
        result = parser.parse_fixtures(raw, league="PL", country="England")
        assert result[0].home_score is None
        assert result[0].away_score is None

    def test_invalid_timestamp_falls_back_to_raw_string(self):
        from src.scrapers.flashscore.parser import FlashScoreParser
        raw = (
            "ZEE\xf7id1\xacAA\xf7Arsenal\xacAB\xf7Chelsea\xac"
            "AD\xf7not-a-timestamp\xacAE\xf7scheduled\xac~"
        )
        parser = FlashScoreParser()
        result = parser.parse_fixtures(raw, league="PL", country="England")
        assert result[0].match_datetime == "not-a-timestamp"

    def test_empty_timestamp_returns_empty_string(self):
        from src.scrapers.flashscore.parser import FlashScoreParser
        raw = (
            "ZEE\xf7id1\xacAA\xf7Arsenal\xacAB\xf7Chelsea\xac"
            "AD\xf7\xacAE\xf7scheduled\xac~"
        )
        parser = FlashScoreParser()
        result = parser.parse_fixtures(raw, league="PL", country="England")
        assert result[0].match_datetime == ""

    def test_multiple_entries_all_parsed(self):
        from src.scrapers.flashscore.parser import FlashScoreParser
        raw = (
            "ZEE\xf7id1\xacAA\xf7Arsenal\xacAB\xf7Chelsea\xacAE\xf7scheduled\xac~"
            "ZEE\xf7id2\xacAA\xf7Liverpool\xacAB\xf7City\xacAE\xf7finished\xacAF\xf71\xacAG\xf70\xac~"
            "ZEE\xf7id3\xacAA\xf7Spurs\xacAB\xf7Everton\xacAE\xf7scheduled\xac~"
        )
        parser = FlashScoreParser()
        result = parser.parse_fixtures(raw, league="PL", country="England")
        assert len(result) == 3

    def test_to_dict_values_are_consistent(self):
        from src.scrapers.flashscore.parser import FlashScoreParser
        raw = (
            "ZEE\xf7id1\xacAA\xf7Arsenal\xacAB\xf7Chelsea\xac"
            "AD\xf71714000000\xacAE\xf7scheduled\xacAF\xf7\xacAG\xf7\xac~"
        )
        parser = FlashScoreParser()
        fixture = parser.parse_fixtures(raw, league="PL", country="England")[0]
        d = fixture.to_dict()
        assert d["home_team"] == fixture.home_team
        assert d["away_team"] == fixture.away_team
        assert d["match_id"] == fixture.match_id
        assert d["status"] == fixture.status


class TestFlashScoreHttpClientEdgeCases:

    def test_token_reset_on_manual_clear(self):
        from src.scrapers.flashscore.http_client import FlashScoreHttpClient
        config = MagicMock()
        config.request_timeout = 10
        config.user_agent = "Mozilla"
        config.base_url = "https://www.flashscore.com"
        config.token_url = "https://www.flashscore.com"

        client = FlashScoreHttpClient(config)
        client._token = "old_token"
        client._token = None
        assert client._token is None

    def test_headers_include_user_agent(self):
        from src.scrapers.flashscore.http_client import FlashScoreHttpClient
        config = MagicMock()
        config.user_agent = "TestAgent/1.0"
        config.base_url = "https://www.flashscore.com"
        config.request_timeout = 5
        config.token_url = "https://www.flashscore.com"
        config.leagues = {}

        client = FlashScoreHttpClient(config)
        headers = client._headers()
        assert headers["User-Agent"] == "TestAgent/1.0"
        assert headers["Referer"] == "https://www.flashscore.com"


class TestFlashScoreConfigIntegration:

    def test_config_loaded_with_flashscore_section(self):
        from config.config_loader import load_config
        config = load_config("config/config.yaml")
        fs = config.scrapers.flashscore
        assert fs.base_url == "https://www.flashscore.com"
        assert fs.http_enabled is True
        assert fs.playwright_fallback_enabled is True
        assert "E0" in fs.leagues
        assert "CL" in fs.leagues

    def test_flashscore_leagues_cover_main_competitions(self):
        from config.config_loader import load_config
        config = load_config("config/config.yaml")
        leagues = config.scrapers.flashscore.leagues
        expected = {"E0", "SP1", "D1", "I1", "F1", "P1", "CL", "EL"}
        assert expected.issubset(set(leagues.keys()))
