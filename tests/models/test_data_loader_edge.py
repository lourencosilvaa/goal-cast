"""Phase 3 edge-case tests for FootballDataLoader HF Parquet integration."""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from config.config_loader import DataConfig, HuggingFaceConfig


@pytest.fixture
def data_config() -> DataConfig:
    return DataConfig(
        base_url="https://www.football-data.co.uk/mmz4281",
        seasons=["2324", "2425"],
        leagues={"E0": "Premier League", "SP1": "La Liga"},
        columns_to_keep=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"],
    )


def _write_league_parquet(tmp_path: Path, league: str, seasons: list[str]) -> Path:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    rows = []
    for season in seasons:
        for i in range(3):
            rows.append({
                "Date": "01/09/2023", "HomeTeam": f"A{i}", "AwayTeam": f"B{i}",
                "FTHG": 1, "FTAG": 0, "FTR": "H", "League": "Premier League", "Season": season,
            })
    pd.DataFrame(rows).to_parquet(datasets_dir / f"{league}.parquet", index=False)
    return datasets_dir


class TestHFParquetEdgeCases:

    def test_hf_disabled_falls_through_to_cache(self, data_config, tmp_path):
        """When no hf_config is provided, HF path is skipped entirely."""
        from src.models.data_loader import FootballDataLoader
        loader = FootballDataLoader(data_config)
        assert loader.hf_config is None
        result = loader._load_from_hf_parquet("E0", "2324")
        assert result.empty

    def test_hf_parquet_corrupt_file_returns_empty(self, data_config, tmp_path):
        """A corrupt Parquet file is silently skipped and returns empty DataFrame."""
        from src.models.data_loader import FootballDataLoader
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        (datasets_dir / "E0.parquet").write_text("not valid parquet content")

        hf_cfg = HuggingFaceConfig(
            repo_id="x", hf_token="", local_dir=str(tmp_path),
            model_filename="m.joblib", dataset_subfolder="datasets",
        )
        loader = FootballDataLoader(data_config, hf_config=hf_cfg)
        result = loader._load_from_hf_parquet("E0", "2324")
        assert result.empty

    def test_hf_parquet_empty_season_slice_returns_empty(self, data_config, tmp_path):
        """Parquet exists but requested season has zero rows."""
        from src.models.data_loader import FootballDataLoader
        _write_league_parquet(tmp_path, "E0", ["2324"])

        hf_cfg = HuggingFaceConfig(
            repo_id="x", hf_token="", local_dir=str(tmp_path),
            model_filename="m.joblib", dataset_subfolder="datasets",
        )
        loader = FootballDataLoader(data_config, hf_config=hf_cfg)
        result = loader._load_from_hf_parquet("E0", "2425")
        assert result.empty

    def test_hf_parquet_missing_required_columns_drops_rows(self, data_config, tmp_path):
        """Rows missing HomeTeam/AwayTeam/FTR are dropped."""
        from src.models.data_loader import FootballDataLoader
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        rows = [
            {"Date": "01/09/2023", "HomeTeam": None, "AwayTeam": "B", "FTR": "H",
             "FTHG": 1, "FTAG": 0, "League": "Premier League", "Season": "2324"},
            {"Date": "01/09/2023", "HomeTeam": "A", "AwayTeam": "B", "FTR": "H",
             "FTHG": 1, "FTAG": 0, "League": "Premier League", "Season": "2324"},
        ]
        pd.DataFrame(rows).to_parquet(datasets_dir / "E0.parquet", index=False)

        hf_cfg = HuggingFaceConfig(
            repo_id="x", hf_token="", local_dir=str(tmp_path),
            model_filename="m.joblib", dataset_subfolder="datasets",
        )
        loader = FootballDataLoader(data_config, hf_config=hf_cfg)
        result = loader._load_from_hf_parquet("E0", "2324")
        assert len(result) == 1
        assert result.iloc[0]["HomeTeam"] == "A"

    def test_load_season_prefers_hf_over_local_cache(self, data_config, tmp_path):
        """HF Parquet takes priority over local CSV cache."""
        from src.models.data_loader import FootballDataLoader
        _write_league_parquet(tmp_path, "E0", ["2324"])

        hf_cfg = HuggingFaceConfig(
            repo_id="x", hf_token="", local_dir=str(tmp_path),
            model_filename="m.joblib", dataset_subfolder="datasets",
        )
        loader = FootballDataLoader(data_config, hf_config=hf_cfg)

        # Patch the web download so we can confirm it is never called
        with patch.object(loader, "_is_cache_valid", return_value=True):
            with patch("pandas.read_csv", side_effect=AssertionError("Should not reach web")):
                result = loader.load_season("E0", "2324")

        assert not result.empty

    def test_load_season_falls_back_when_hf_dir_missing(self, data_config, tmp_path):
        """If HF local_dir does not have datasets/, fall through to cache/web."""
        from src.models.data_loader import FootballDataLoader
        hf_cfg = HuggingFaceConfig(
            repo_id="x", hf_token="", local_dir=str(tmp_path),
            model_filename="m.joblib", dataset_subfolder="datasets",
        )
        loader = FootballDataLoader(data_config, hf_config=hf_cfg)
        result = loader._load_from_hf_parquet("E0", "2324")
        assert result.empty  # no datasets dir → empty → falls through

    def test_save_as_parquet_unknown_league_uses_name_as_code(self, data_config, tmp_path):
        """League names not in config are saved with the name as the filename."""
        from src.models.data_loader import FootballDataLoader
        loader = FootballDataLoader(data_config)

        df = pd.DataFrame([{
            "Date": "01/09/2023", "HomeTeam": "X", "AwayTeam": "Y",
            "FTHG": 0, "FTAG": 1, "FTR": "A", "League": "Unknown League", "Season": "2324",
        }])
        out_dir = tmp_path / "parquet_out"
        loader.save_as_parquet(df, out_dir)
        # File should be named after the league name since there's no code mapping
        assert (out_dir / "Unknown League.parquet").exists()

    def test_save_as_parquet_raises_without_league_column(self, data_config, tmp_path):
        """ValueError is raised when DataFrame has no League column."""
        from src.models.data_loader import FootballDataLoader
        loader = FootballDataLoader(data_config)

        df = pd.DataFrame([{"HomeTeam": "A", "AwayTeam": "B", "FTR": "H"}])
        with pytest.raises(ValueError, match="League"):
            loader.save_as_parquet(df, tmp_path / "out")

    def test_save_as_parquet_multiple_seasons_in_one_file(self, data_config, tmp_path):
        """All seasons for a league are saved in one Parquet file with Season column."""
        from src.models.data_loader import FootballDataLoader
        loader = FootballDataLoader(data_config)

        rows = [
            {"Date": "01/09/2023", "HomeTeam": "A", "AwayTeam": "B",
             "FTHG": 1, "FTAG": 0, "FTR": "H", "League": "Premier League", "Season": "2324"},
            {"Date": "01/09/2024", "HomeTeam": "C", "AwayTeam": "D",
             "FTHG": 0, "FTAG": 1, "FTR": "A", "League": "Premier League", "Season": "2425"},
        ]
        out_dir = tmp_path / "out"
        loader.save_as_parquet(pd.DataFrame(rows), out_dir)

        loaded = pd.read_parquet(out_dir / "E0.parquet")
        assert len(loaded) == 2
        assert set(loaded["Season"].unique()) == {"2324", "2425"}


class TestLoadSeasonCacheAndWebPaths:

    def test_cached_path_returns_expected_path(self, data_config):
        from src.models.data_loader import FootballDataLoader, _CACHE_DIR
        loader = FootballDataLoader(data_config)
        result = loader._cached_path("E0", "2324")
        assert result == _CACHE_DIR / "2324_E0.csv"

    def test_is_cache_valid_false_when_file_missing(self, data_config, tmp_path):
        from src.models.data_loader import FootballDataLoader
        loader = FootballDataLoader(data_config)
        assert not loader._is_cache_valid(tmp_path / "nonexistent.csv")

    def test_is_cache_valid_true_when_file_fresh(self, data_config, tmp_path):
        from src.models.data_loader import FootballDataLoader
        csv_file = tmp_path / "fresh.csv"
        csv_file.write_text("HomeTeam,AwayTeam,FTR\nA,B,H\n")
        loader = FootballDataLoader(data_config)
        assert loader._is_cache_valid(csv_file)

    def test_is_cache_valid_false_when_file_stale(self, data_config, tmp_path):
        import os
        from src.models.data_loader import FootballDataLoader
        csv_file = tmp_path / "stale.csv"
        csv_file.write_text("HomeTeam,AwayTeam,FTR\nA,B,H\n")
        old_time = os.path.getmtime(csv_file) - (25 * 60 * 60)  # 25h ago
        os.utime(csv_file, (old_time, old_time))
        loader = FootballDataLoader(data_config)
        assert not loader._is_cache_valid(csv_file)

    def test_load_season_from_local_cache(self, data_config, tmp_path, monkeypatch):
        """load_season reads from local CSV cache when HF is absent and cache is valid."""
        from src.models.data_loader import FootballDataLoader, _CACHE_DIR
        import pandas as pd

        # Write a fake cache CSV
        cache_file = _CACHE_DIR / "2324_E0.csv"
        rows = pd.DataFrame([
            {"HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
             "FTHG": 2, "FTAG": 0, "FTR": "H", "Date": "01/09/2023",
             "HTAG": 0, "HTHG": 1, "HTR": "H"},
        ])
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rows.to_csv(cache_file, index=False)

        loader = FootballDataLoader(data_config)
        monkeypatch.setattr(loader, "_is_cache_valid", lambda p: True)
        df = loader.load_season("E0", "2324")

        assert not df.empty
        assert df.iloc[0]["HomeTeam"] == "Arsenal"
        assert df.iloc[0]["Season"] == "2324"
        assert df.iloc[0]["League"] == "Premier League"

        cache_file.unlink(missing_ok=True)

    def test_load_season_web_fallback_on_error(self, data_config, monkeypatch):
        """When cache is invalid and web fails, returns empty DataFrame."""
        from src.models.data_loader import FootballDataLoader

        loader = FootballDataLoader(data_config)
        monkeypatch.setattr(loader, "_is_cache_valid", lambda p: False)

        import pandas as pd
        monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: (_ for _ in ()).throw(Exception("network error")))

        df = loader.load_season("E0", "2324")
        assert df.empty

    def test_load_season_web_success(self, data_config, monkeypatch):
        """load_season successfully downloads from web when cache is invalid."""
        import pandas as pd
        from unittest.mock import MagicMock
        from src.models.data_loader import FootballDataLoader

        loader = FootballDataLoader(data_config)
        monkeypatch.setattr(loader, "_is_cache_valid", lambda p: False)

        web_df = pd.DataFrame([{
            "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
            "FTHG": 2, "FTAG": 0, "FTR": "H", "Date": "01/09/2023",
        }])

        read_calls: list[str] = []

        def fake_read_csv(path_or_url, *args, **kwargs):
            read_calls.append(str(path_or_url))
            return web_df.copy()

        save_calls: list[str] = []

        monkeypatch.setattr(pd, "read_csv", fake_read_csv)

        original_to_csv = pd.DataFrame.to_csv
        def fake_to_csv(self_df, path, **kwargs):
            save_calls.append(str(path))

        monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)

        df = loader.load_season("E0", "2324")

        assert not df.empty
        assert df.iloc[0]["HomeTeam"] == "Arsenal"
        assert df.iloc[0]["League"] == "Premier League"
        assert df.iloc[0]["Season"] == "2324"
        assert len(save_calls) == 1  # CSV was saved to cache

    def test_load_season_cache_corrupt_falls_through_to_web(self, data_config, monkeypatch):
        """When local cache CSV is corrupt, load_season falls through to web."""
        import pandas as pd
        from src.models.data_loader import FootballDataLoader

        loader = FootballDataLoader(data_config)
        monkeypatch.setattr(loader, "_is_cache_valid", lambda p: True)

        call_count = [0]

        def fake_read_csv(path_or_url, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("corrupt cache")
            raise Exception("web also failed")

        monkeypatch.setattr(pd, "read_csv", fake_read_csv)
        df = loader.load_season("E0", "2324")
        assert df.empty
        assert call_count[0] == 2  # tried cache then web

    def test_load_all_aggregates_seasons(self, data_config, monkeypatch):
        """load_all calls load_season for each league/season combination."""
        import pandas as pd
        from src.models.data_loader import FootballDataLoader

        call_log: list[tuple[str, str]] = []

        def fake_load_season(league: str, season: str) -> pd.DataFrame:
            call_log.append((league, season))
            return pd.DataFrame([
                {"HomeTeam": "A", "AwayTeam": "B", "FTR": "H",
                 "League": data_config.leagues.get(league, league), "Season": season},
            ])

        loader = FootballDataLoader(data_config)
        monkeypatch.setattr(loader, "load_season", fake_load_season)
        result = loader.load_all()

        expected_calls = len(data_config.leagues) * len(data_config.seasons)
        assert len(call_log) == expected_calls
        assert not result.empty

    def test_load_all_returns_empty_when_all_fail(self, data_config, monkeypatch):
        """load_all returns empty DataFrame when all seasons fail to load."""
        import pandas as pd
        from src.models.data_loader import FootballDataLoader

        loader = FootballDataLoader(data_config)
        monkeypatch.setattr(loader, "load_season", lambda l, s: pd.DataFrame())
        result = loader.load_all()
        assert result.empty

    def test_load_from_csv_filters_columns(self, data_config, tmp_path):
        """load_from_csv applies columns_to_keep and drops rows with missing required fields."""
        import pandas as pd
        from src.models.data_loader import FootballDataLoader

        csv_file = tmp_path / "test.csv"
        pd.DataFrame([
            {"HomeTeam": "A", "AwayTeam": "B", "FTR": "H",
             "FTHG": 1, "FTAG": 0, "Date": "01/09/2023", "ExtraCol": "ignore"},
            {"HomeTeam": None, "AwayTeam": "B", "FTR": "H",
             "FTHG": 0, "FTAG": 0, "Date": "02/09/2023", "ExtraCol": "ignore"},
        ]).to_csv(csv_file, index=False)

        loader = FootballDataLoader(data_config)
        df = loader.load_from_csv(str(csv_file))

        assert "ExtraCol" not in df.columns
        assert len(df) == 1  # row with None HomeTeam dropped


class TestUploadScriptParquetBuilding:

    def test_build_parquet_datasets_groups_by_league(self, tmp_path):
        """_build_parquet_datasets creates one Parquet per league."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2]))
        from scripts.upload_to_hf import _build_parquet_datasets

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Write two seasons for two leagues
        for season in ["2324", "2425"]:
            for league in ["E0", "SP1"]:
                rows = [{"HomeTeam": "A", "AwayTeam": "B", "FTR": "H", "FTHG": 1, "FTAG": 0}]
                pd.DataFrame(rows).to_csv(cache_dir / f"{season}_{league}.csv", index=False)

        out_tmp = tmp_path / "out"
        out_tmp.mkdir()
        result = _build_parquet_datasets(cache_dir, out_tmp)

        assert "E0" in result
        assert "SP1" in result
        assert result["E0"].exists()
        assert result["SP1"].exists()

    def test_build_parquet_datasets_merges_seasons(self, tmp_path):
        """Each Parquet file contains all seasons for that league."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2]))
        from scripts.upload_to_hf import _build_parquet_datasets

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        for season in ["2324", "2425"]:
            rows = [{"HomeTeam": f"T{season}", "AwayTeam": "B", "FTR": "H"}]
            pd.DataFrame(rows).to_csv(cache_dir / f"{season}_E0.csv", index=False)

        out_tmp = tmp_path / "out"
        out_tmp.mkdir()
        result = _build_parquet_datasets(cache_dir, out_tmp)

        combined = pd.read_parquet(result["E0"])
        assert len(combined) == 2
        assert set(combined["Season"].unique()) == {"2324", "2425"}

    def test_build_parquet_datasets_empty_cache_returns_empty(self, tmp_path):
        """Empty cache directory produces no Parquet files."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2]))
        from scripts.upload_to_hf import _build_parquet_datasets

        cache_dir = tmp_path / "empty_cache"
        cache_dir.mkdir()
        out_tmp = tmp_path / "out"
        out_tmp.mkdir()

        result = _build_parquet_datasets(cache_dir, out_tmp)
        assert result == {}
