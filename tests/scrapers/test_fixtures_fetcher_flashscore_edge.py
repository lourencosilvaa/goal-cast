"""Edge case tests for FlashScore fallback in fixtures_fetcher."""

from unittest.mock import MagicMock, patch

import pytest

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
        mock_scraper.scrape_fixtures.return_value = [_make_fs_fixture(dt="")]
        mock_build.return_value = mock_scraper

        result = _fetch_flashscore_fixtures("28/04/2024", leagues=["E0"])

        assert result[0].time == ""

    @patch("src.scrapers.fixtures_fetcher._build_flashscore_scraper")
    def test_unknown_league_code_uses_flashscore_league_name(self, mock_build):
        mock_scraper = MagicMock()
        fs = _make_fs_fixture()
        fs = FlashScoreFixture(
            match_id="id1", home_team="Arsenal", away_team="Chelsea",
            league="Bundesliga", country="Germany",
            match_datetime="", status="scheduled",
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


class TestInferenceServiceEdgeCases:

    @patch("src.backend.services.inference_service.requests.post")
    def test_space_500_raises_http_error(self, mock_post):
        import requests as req
        from src.backend.services.inference_service import InferenceService

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req.exceptions.HTTPError("500")
        mock_post.return_value = mock_response

        config = MagicMock()
        config.inference.enabled = True
        config.inference.space_url = "https://user.hf.space"

        svc = InferenceService(config)

        with pytest.raises(req.exceptions.HTTPError):
            svc.run(target_date="28/04/2024")

    def test_disabled_inference_raises_runtime_error(self):
        from src.backend.services.inference_service import InferenceService

        config = MagicMock()
        config.inference.enabled = False
        svc = InferenceService(config)

        with pytest.raises(RuntimeError):
            svc.run()


class TestNvidiaKeyEdgeCases:

    def test_get_nvidia_key_uses_nvidia_service(self):
        from src.backend.services.api_key_service import ApiKeyService, _SERVICE_NVIDIA
        svc = ApiKeyService(supabase=MagicMock(), encryption=MagicMock())
        svc._db.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(data=None)

        result = svc.get_user_key("user-1", service=_SERVICE_NVIDIA)

        assert result is None

    def test_set_nvidia_key_empty_raises(self):
        from src.backend.services.api_key_service import ApiKeyService, _SERVICE_NVIDIA
        import pytest

        svc = ApiKeyService(supabase=MagicMock(), encryption=MagicMock())

        with pytest.raises(ValueError):
            svc.set_user_key("user-1", "", service=_SERVICE_NVIDIA)

    def test_gemini_default_unchanged(self):
        from src.backend.services.api_key_service import ApiKeyService

        mock_db = MagicMock()
        mock_db.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(data=None)
        svc = ApiKeyService(supabase=mock_db, encryption=MagicMock())

        svc.get_user_key("user-1")

        second_eq_calls = str(
            mock_db.table.return_value.select.return_value.eq.return_value.eq.call_args_list
        )
        assert "gemini" in second_eq_calls
