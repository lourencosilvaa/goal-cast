"""Tests for FlashScore fallback integration in fixtures_fetcher."""

from unittest.mock import MagicMock, patch

from src.scrapers.fixtures_fetcher import _fetch_flashscore_fixtures
from src.scrapers.base_scraper import FlashScoreFixture


def _make_fs_fixture(home: str = "Arsenal", away: str = "Chelsea") -> FlashScoreFixture:
    return FlashScoreFixture(
        match_id="id1",
        home_team=home,
        away_team=away,
        league="Premier League",
        country="England",
        match_datetime="2024-04-28T15:00:00+00:00",
        status="scheduled",
        home_score=None,
        away_score=None,
        source_url="https://www.flashscore.com/match/id1/",
    )


class TestFetchFlashscoreFixtures:

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_returns_fixture_list(self, mock_build):
        mock_scraper = MagicMock()
        mock_scraper.scrape_fixtures.return_value = [_make_fs_fixture()]
        mock_build.return_value = mock_scraper

        result = _fetch_flashscore_fixtures("28/04/2024", leagues=["E0"])

        assert isinstance(result, list)
        assert len(result) == 1

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_converts_to_fixture_dataclass(self, mock_build):
        from src.scrapers.fixtures_fetcher import Fixture

        mock_scraper = MagicMock()
        mock_scraper.scrape_fixtures.return_value = [_make_fs_fixture()]
        mock_build.return_value = mock_scraper

        result = _fetch_flashscore_fixtures("28/04/2024", leagues=["E0"])

        assert isinstance(result[0], Fixture)

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_converts_team_names(self, mock_build):
        mock_scraper = MagicMock()
        mock_scraper.scrape_fixtures.return_value = [
            _make_fs_fixture("Liverpool", "Man City")
        ]
        mock_build.return_value = mock_scraper

        result = _fetch_flashscore_fixtures("28/04/2024", leagues=["E0"])

        assert result[0].home_team == "Liverpool"
        assert result[0].away_team == "Man City"

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_b365_odds_are_zero_for_flashscore_source(self, mock_build):
        mock_scraper = MagicMock()
        mock_scraper.scrape_fixtures.return_value = [_make_fs_fixture()]
        mock_build.return_value = mock_scraper

        result = _fetch_flashscore_fixtures("28/04/2024", leagues=["E0"])

        assert result[0].b365_home == 0.0
        assert result[0].b365_draw == 0.0
        assert result[0].b365_away == 0.0

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_filters_by_league(self, mock_build):
        mock_scraper = MagicMock()
        mock_scraper.scrape_fixtures.return_value = [_make_fs_fixture()]
        mock_build.return_value = mock_scraper

        _fetch_flashscore_fixtures("28/04/2024", leagues=["E0"])

        mock_scraper.scrape_fixtures.assert_called_with("E0")

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_returns_empty_on_scraper_error(self, mock_build):
        from src.scrapers.flashscore.exceptions import FlashScoreUnavailableError

        mock_scraper = MagicMock()
        mock_scraper.scrape_fixtures.side_effect = FlashScoreUnavailableError("down")
        mock_build.return_value = mock_scraper

        result = _fetch_flashscore_fixtures("28/04/2024", leagues=["E0"])

        assert result == []

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_none_leagues_scrapes_all_supported(self, mock_build):
        mock_scraper = MagicMock()
        mock_scraper.get_available_leagues.return_value = {
            "E0": "england/premier-league",
            "SP1": "spain/laliga",
        }
        mock_scraper.scrape_fixtures.return_value = []
        mock_build.return_value = mock_scraper

        _fetch_flashscore_fixtures("28/04/2024", leagues=None)

        assert mock_scraper.scrape_fixtures.call_count == 2

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_date_is_passed_to_fixture(self, mock_build):
        mock_scraper = MagicMock()
        mock_scraper.scrape_fixtures.return_value = [_make_fs_fixture()]
        mock_build.return_value = mock_scraper

        result = _fetch_flashscore_fixtures("28/04/2024", leagues=["E0"])

        assert result[0].date == "28/04/2024"
