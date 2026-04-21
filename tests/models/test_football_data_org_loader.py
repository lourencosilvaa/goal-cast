import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.football_data_org_loader import FootballDataOrgLoader, OrgLoaderConfig


def _make_config(api_key: str = "test-api-key") -> OrgLoaderConfig:
    return OrgLoaderConfig(
        api_key=api_key,
        base_url="https://api.football-data.org/v4",
        request_timeout=10,
        competitions={
            "CL": "Champions League",
            "EL": "Europa League",
        },
    )


SAMPLE_MATCHES_RESPONSE = {
    "matches": [
        {
            "utcDate": "2025-03-15T20:00:00Z",
            "homeTeam": {"name": "Real Madrid"},
            "awayTeam": {"name": "Bayern Munich"},
            "score": {
                "fullTime": {"home": 2, "away": 1},
                "halfTime": {"home": 1, "away": 0},
            },
            "status": "FINISHED",
        },
        {
            "utcDate": "2025-03-16T20:00:00Z",
            "homeTeam": {"name": "PSG"},
            "awayTeam": {"name": "Arsenal"},
            "score": {
                "fullTime": {"home": None, "away": None},
                "halfTime": {"home": None, "away": None},
            },
            "status": "SCHEDULED",
        },
    ]
}


class TestFootballDataOrgLoader:

    def test_raises_config_error_when_api_key_missing(self):
        from src.models.football_data_org_loader import ConfigurationError
        with pytest.raises(ConfigurationError, match="api_key"):
            FootballDataOrgLoader(_make_config(api_key=""))

    def test_load_season_returns_dataframe_with_required_columns(self, tmp_path):
        config = _make_config()
        loader = FootballDataOrgLoader(config, cache_dir=str(tmp_path))

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = SAMPLE_MATCHES_RESPONSE
            mock_get.return_value = mock_response

            df = loader.load_season(competition="CL", season="2425")

        assert not df.empty
        for col in ["HomeTeam", "AwayTeam", "Date", "League"]:
            assert col in df.columns

    def test_load_season_filters_only_finished_matches(self, tmp_path):
        config = _make_config()
        loader = FootballDataOrgLoader(config, cache_dir=str(tmp_path))

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = SAMPLE_MATCHES_RESPONSE
            mock_get.return_value = mock_response

            df = loader.load_season(competition="CL", season="2425")

        assert len(df) == 1
        assert df.iloc[0]["HomeTeam"] == "Real Madrid"

    def test_load_season_raises_on_api_error(self, tmp_path):
        config = _make_config()
        loader = FootballDataOrgLoader(config, cache_dir=str(tmp_path))

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.raise_for_status.side_effect = Exception("Rate limit")
            mock_get.return_value = mock_response

            with pytest.raises(Exception):
                loader.load_season(competition="CL", season="2425")

    def test_load_season_uses_cache_on_second_call(self, tmp_path):
        config = _make_config()
        loader = FootballDataOrgLoader(config, cache_dir=str(tmp_path))

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = SAMPLE_MATCHES_RESPONSE
            mock_get.return_value = mock_response

            loader.load_season(competition="CL", season="2425")
            loader.load_season(competition="CL", season="2425")

        assert mock_get.call_count == 1

    def test_supported_competitions_from_config(self):
        config = _make_config()
        loader = FootballDataOrgLoader(config)
        comps = loader.supported_competitions()
        assert "CL" in comps
        assert "EL" in comps
