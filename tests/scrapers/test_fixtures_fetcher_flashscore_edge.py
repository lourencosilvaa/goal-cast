"""Edge case tests for FlashScore fallback in fixtures_fetcher."""

from unittest.mock import MagicMock, patch

from src.scrapers.fixtures_fetcher import _fetch_flashscore_fixtures, fetch_fixtures
from src.scrapers.base_scraper import FlashScoreFixture


def _make_fs_fixture(home: str = "Arsenal", away: str = "Chelsea",
                     dt: str = "2024-04-28T15:00:00+00:00") -> FlashScoreFixture:
    return FlashScoreFixture(
        match_id="id1", home_team=home, away_team=away,
        league="Premier League", country="England",
        match_datetime=dt, status="scheduled",
        home_score=None, away_score=None,
        source_url="https://www.flashscore.com/match/id1/",
    )


class TestFlashScoreFallbackEdgeCases:

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_time_extracted_from_datetime(self, mock_build):
        mock_scraper = MagicMock()
        mock_scraper.scrape_fixtures.return_value = [
            _make_fs_fixture(dt="2024-04-28T15:30:00+00:00")
        ]
        mock_build.return_value = mock_scraper

        result = _fetch_flashscore_fixtures("28/04/2024", leagues=["E0"])

        assert result[0].time == "15:30"

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_short_datetime_gives_empty_time(self, mock_build):
        mock_scraper = MagicMock()
        # datetime too short to extract HH:MM but still matches the target date
        mock_scraper.scrape_fixtures.return_value = [_make_fs_fixture(dt="2024-04-28")]
        mock_build.return_value = mock_scraper

        result = _fetch_flashscore_fixtures("28/04/2024", leagues=["E0"])

        assert result[0].time == ""

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_unknown_league_code_uses_flashscore_league_name(self, mock_build):
        mock_scraper = MagicMock()
        fs = FlashScoreFixture(
            match_id="id1", home_team="Arsenal", away_team="Chelsea",
            league="Bundesliga", country="Germany",
            match_datetime="2024-04-28T15:00:00", status="scheduled",
            home_score=None, away_score=None,
            source_url="",
        )
        mock_scraper.scrape_fixtures.return_value = [fs]
        mock_build.return_value = mock_scraper

        result = _fetch_flashscore_fixtures("28/04/2024", leagues=["D1"])

        assert result[0].league == "Bundesliga"

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_multiple_leagues_aggregated(self, mock_build):
        mock_scraper = MagicMock()
        mock_scraper.scrape_fixtures.side_effect = [
            [_make_fs_fixture("Arsenal", "Chelsea")],
            [_make_fs_fixture("Barcelona", "Real Madrid")],
        ]
        mock_build.return_value = mock_scraper

        result = _fetch_flashscore_fixtures("28/04/2024", leagues=["E0", "SP1"])

        assert len(result) == 2

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_build_error_returns_empty(self, mock_build):
        mock_build.side_effect = Exception("config error")

        result = _fetch_flashscore_fixtures("28/04/2024", leagues=["E0"])

        assert result == []

    @patch("src.scrapers.fixtures_fetcher._fetch_flashscore_fixtures")
    @patch("src.scrapers.fixtures_fetcher._fetch_odds_api_fixtures")
    @patch("src.scrapers.fixtures_fetcher._get_fixtures_csv_text")
    def test_flashscore_called_only_when_csv_and_oddsapi_empty(
        self, mock_csv, mock_oddsapi, mock_fs
    ):
        mock_csv.return_value = "Div,Date,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
        mock_oddsapi.return_value = []
        mock_fs.return_value = []

        fetch_fixtures(target_date="28/04/2024")

        mock_fs.assert_called_once()

    @patch("src.scrapers.fixtures_fetcher._fetch_flashscore_fixtures")
    @patch("src.scrapers.fixtures_fetcher._fetch_odds_api_fixtures")
    @patch("src.scrapers.fixtures_fetcher._get_fixtures_csv_text")
    def test_flashscore_not_called_when_csv_has_fixtures(
        self, mock_csv, mock_oddsapi, mock_fs
    ):
        mock_csv.return_value = (
            "Div,Date,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
            "E0,28/04/2024,Arsenal,Chelsea,1.8,3.5,4.2\n"
        )
        mock_oddsapi.return_value = []
        mock_fs.return_value = []

        fetch_fixtures(target_date="28/04/2024")

        mock_fs.assert_not_called()

