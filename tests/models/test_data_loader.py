"""Phase 1 + Phase 3 tests for FootballDataLoader HF Parquet integration."""

import pandas as pd
import pytest
from pathlib import Path

from config.config_loader import DataConfig, HuggingFaceConfig


@pytest.fixture
def data_config() -> DataConfig:
    return DataConfig(
        base_url="https://www.football-data.co.uk/mmz4281",
        seasons=["2324", "2425"],
        leagues={"E0": "Premier League", "SP1": "La Liga"},
        served_leagues=["E0", "SP1"],
        columns_to_keep=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"],
    )


@pytest.fixture
def hf_config(tmp_path: Path) -> HuggingFaceConfig:
    return HuggingFaceConfig(
        repo_id="test/repo",
        hf_token="",
        local_dir=str(tmp_path),
        model_filename="ensemble_model.joblib",
        dataset_subfolder="datasets",
    )


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Write a minimal per-league Parquet file with two seasons."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    rows = []
    for season in ["2324", "2425"]:
        for i in range(5):
            rows.append(
                {
                    "Date": "01/09/2023",
                    "HomeTeam": f"TeamA_{i}",
                    "AwayTeam": f"TeamB_{i}",
                    "FTHG": 1,
                    "FTAG": 0,
                    "FTR": "H",
                    "League": "Premier League",
                    "Season": season,
                }
            )

    df = pd.DataFrame(rows)
    df.to_parquet(datasets_dir / "E0.parquet", index=False)
    return datasets_dir


class TestHuggingFaceConfigExists:

    def test_hf_config_class_importable(self):
        from config.config_loader import HuggingFaceConfig
        assert HuggingFaceConfig is not None

    def test_hf_config_has_dataset_subfolder(self):
        from config.config_loader import HuggingFaceConfig
        cfg = HuggingFaceConfig(
            repo_id="x/y",
            hf_token="",
            local_dir="/tmp",
            model_filename="model.joblib",
            dataset_subfolder="datasets",
        )
        assert cfg.dataset_subfolder == "datasets"

    def test_hf_config_dataset_subfolder_default(self):
        from config.config_loader import HuggingFaceConfig
        cfg = HuggingFaceConfig(repo_id="", hf_token="", local_dir="/tmp", model_filename="m.joblib")
        assert cfg.dataset_subfolder == "datasets"


class TestConfigHasHuggingFaceField:

    def test_config_has_huggingface_attr(self, config_path: Path):
        from config.config_loader import load_config
        config = load_config(config_path)
        assert hasattr(config, "huggingface")

    def test_config_huggingface_has_dataset_subfolder(self, config_path: Path):
        from config.config_loader import load_config
        config = load_config(config_path)
        assert hasattr(config.huggingface, "dataset_subfolder")
        assert config.huggingface.dataset_subfolder != ""


class TestDataLoaderAcceptsHFConfig:

    def test_loader_accepts_hf_config_param(self, data_config, hf_config):
        from src.models.data_loader import FootballDataLoader
        loader = FootballDataLoader(data_config, hf_config=hf_config)
        assert loader is not None

    def test_loader_works_without_hf_config(self, data_config):
        from src.models.data_loader import FootballDataLoader
        loader = FootballDataLoader(data_config)
        assert loader is not None


class TestLoadFromHFParquet:

    def test_load_season_from_hf_parquet(self, data_config, hf_config, sample_parquet, tmp_path):
        from src.models.data_loader import FootballDataLoader
        hf_config = HuggingFaceConfig(
            repo_id="test/repo",
            hf_token="",
            local_dir=str(tmp_path),
            model_filename="ensemble_model.joblib",
            dataset_subfolder="datasets",
        )
        loader = FootballDataLoader(data_config, hf_config=hf_config)
        df = loader.load_season("E0", "2324")
        assert not df.empty
        assert "HomeTeam" in df.columns
        assert "FTR" in df.columns
        assert all(df["Season"] == "2324")

    def test_load_season_hf_filters_correct_season(self, data_config, tmp_path):
        from src.models.data_loader import FootballDataLoader
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()

        rows_2324 = [
            {"Date": "01/09/2023", "HomeTeam": f"A{i}", "AwayTeam": f"B{i}",
             "FTHG": 1, "FTAG": 0, "FTR": "H", "League": "Premier League", "Season": "2324"}
            for i in range(3)
        ]
        rows_2425 = [
            {"Date": "01/09/2024", "HomeTeam": f"C{i}", "AwayTeam": f"D{i}",
             "FTHG": 0, "FTAG": 1, "FTR": "A", "League": "Premier League", "Season": "2425"}
            for i in range(3)
        ]
        pd.DataFrame(rows_2324 + rows_2425).to_parquet(datasets_dir / "E0.parquet", index=False)

        hf_cfg = HuggingFaceConfig(
            repo_id="x", hf_token="", local_dir=str(tmp_path),
            model_filename="m.joblib", dataset_subfolder="datasets",
        )
        loader = FootballDataLoader(data_config, hf_config=hf_cfg)
        df = loader.load_season("E0", "2324")
        assert not df.empty
        assert all(df["Season"] == "2324")
        assert len(df) == 3

    def test_load_season_returns_empty_when_parquet_missing(self, data_config, tmp_path):
        from src.models.data_loader import FootballDataLoader
        hf_cfg = HuggingFaceConfig(
            repo_id="x", hf_token="", local_dir=str(tmp_path),
            model_filename="m.joblib", dataset_subfolder="datasets",
        )
        loader = FootballDataLoader(data_config, hf_config=hf_cfg)
        # No Parquet files, no web (season that won't exist) — HF path returns empty
        # We only test that HF lookup doesn't crash; it falls back gracefully
        # Without mocking web, we just verify the method is callable
        result = loader._load_from_hf_parquet("E0", "2324")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_load_from_hf_parquet_missing_season_returns_empty(self, data_config, tmp_path):
        from src.models.data_loader import FootballDataLoader
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        rows = [
            {"Date": "01/09/2023", "HomeTeam": "A", "AwayTeam": "B",
             "FTHG": 1, "FTAG": 0, "FTR": "H", "League": "PL", "Season": "2324"}
        ]
        pd.DataFrame(rows).to_parquet(datasets_dir / "E0.parquet", index=False)

        hf_cfg = HuggingFaceConfig(
            repo_id="x", hf_token="", local_dir=str(tmp_path),
            model_filename="m.joblib", dataset_subfolder="datasets",
        )
        loader = FootballDataLoader(data_config, hf_config=hf_cfg)
        df = loader._load_from_hf_parquet("E0", "9999")
        assert df.empty


class TestParquetWithoutLeagueColumn:
    """Regression: upload_to_hf builds Parquet straight from raw CSVs, which
    carry no ``League`` column. The HF Parquet path must fill it in from config
    like the CSV-cache and web paths do — otherwise every League-aware consumer
    (e.g. the Space's /teams endpoint) silently sees nothing."""

    @staticmethod
    def _write_leagueless_parquet(tmp_path: Path, league: str = "E0") -> Path:
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        rows = [
            {"Date": "01/09/2023", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
             "FTHG": 1, "FTAG": 0, "FTR": "H", "Season": "2324"}
        ]
        pd.DataFrame(rows).to_parquet(datasets_dir / f"{league}.parquet", index=False)
        return datasets_dir

    @staticmethod
    def _hf_cfg(tmp_path: Path) -> HuggingFaceConfig:
        return HuggingFaceConfig(
            repo_id="x", hf_token="", local_dir=str(tmp_path),
            model_filename="m.joblib", dataset_subfolder="datasets",
        )

    def test_league_column_is_added_from_config(self, data_config, tmp_path):
        from src.models.data_loader import FootballDataLoader
        self._write_leagueless_parquet(tmp_path)
        loader = FootballDataLoader(data_config, hf_config=self._hf_cfg(tmp_path))
        df = loader._load_from_hf_parquet("E0", "2324")
        assert "League" in df.columns
        assert all(df["League"] == "Premier League")

    def test_season_column_is_preserved(self, data_config, tmp_path):
        from src.models.data_loader import FootballDataLoader
        self._write_leagueless_parquet(tmp_path)
        loader = FootballDataLoader(data_config, hf_config=self._hf_cfg(tmp_path))
        df = loader._load_from_hf_parquet("E0", "2324")
        assert all(df["Season"] == "2324")

    def test_existing_league_values_are_not_overwritten(self, data_config, tmp_path):
        from src.models.data_loader import FootballDataLoader
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        rows = [
            {"Date": "01/09/2023", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
             "FTHG": 1, "FTAG": 0, "FTR": "H", "League": "Custom Name",
             "Season": "2324"}
        ]
        pd.DataFrame(rows).to_parquet(datasets_dir / "E0.parquet", index=False)
        loader = FootballDataLoader(data_config, hf_config=self._hf_cfg(tmp_path))
        df = loader._load_from_hf_parquet("E0", "2324")
        assert all(df["League"] == "Custom Name")


class TestSaveAsParquet:

    def test_save_as_parquet_creates_files(self, data_config, tmp_path):
        from src.models.data_loader import FootballDataLoader
        loader = FootballDataLoader(data_config)

        df = pd.DataFrame([
            {"Date": "01/09/2023", "HomeTeam": "A", "AwayTeam": "B",
             "FTHG": 1, "FTAG": 0, "FTR": "H", "League": "Premier League", "Season": "2324"},
            {"Date": "02/09/2023", "HomeTeam": "C", "AwayTeam": "D",
             "FTHG": 0, "FTAG": 0, "FTR": "D", "League": "La Liga", "Season": "2324"},
        ])

        out_dir = tmp_path / "parquet_out"
        loader.save_as_parquet(df, out_dir)

        assert (out_dir / "E0.parquet").exists()
        assert (out_dir / "SP1.parquet").exists()

    def test_save_as_parquet_round_trip(self, data_config, tmp_path):
        from src.models.data_loader import FootballDataLoader
        loader = FootballDataLoader(data_config)

        df = pd.DataFrame([
            {"Date": "01/09/2023", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
             "FTHG": 2, "FTAG": 1, "FTR": "H", "League": "Premier League", "Season": "2324"},
        ])

        out_dir = tmp_path / "parquet_out"
        loader.save_as_parquet(df, out_dir)

        loaded = pd.read_parquet(out_dir / "E0.parquet")
        assert len(loaded) == 1
        assert loaded.iloc[0]["HomeTeam"] == "Arsenal"
        assert loaded.iloc[0]["Season"] == "2324"


class TestSeasonsConfig:

    def test_config_includes_1819_season(self, config_path: Path):
        from config.config_loader import load_config
        config = load_config(config_path)
        assert "1819" in config.data.seasons

    def test_config_includes_1920_season(self, config_path: Path):
        from config.config_loader import load_config
        config = load_config(config_path)
        assert "1920" in config.data.seasons

    def test_config_includes_2526_season(self, config_path: Path):
        from config.config_loader import load_config
        config = load_config(config_path)
        assert "2526" in config.data.seasons

    def test_config_has_at_least_8_seasons(self, config_path: Path):
        from config.config_loader import load_config
        config = load_config(config_path)
        assert len(config.data.seasons) >= 8


class TestSpaceDataLoaderStaysInSync:
    """``hf_space/src/models/data_loader.py`` is a vendored copy pushed to the
    HuggingFace Space by deploy-hf-space.yml. The League back-fill has to exist
    in both copies or the Space's /teams endpoint silently returns {}."""

    @staticmethod
    def _space_loader_source() -> str:
        path = (
            Path(__file__).resolve().parents[2]
            / "hf_space"
            / "src"
            / "models"
            / "data_loader.py"
        )
        return path.read_text(encoding="utf-8")

    def test_space_copy_exists(self):
        assert self._space_loader_source()

    def test_space_copy_backfills_league_column(self):
        assert 'if "League" not in df.columns:' in self._space_loader_source()

    def test_space_copy_uses_configured_league_names(self):
        assert (
            "df.assign(League=self.config.leagues.get(league, league))"
            in self._space_loader_source()
        )


class TestDownloadedDivisionIsVerified:
    """football-data.co.uk 301-redirects a not-yet-published season's file onto
    a *different* division (``2627/SP1.csv`` → ``2627/SC1.csv``). Trusting the
    response labels Scottish Championship matches as La Liga, poisoning both
    the training corpus and the teams registry, so the payload must be checked
    against the division that was actually requested.
    """

    HEADER_COLUMNS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

    def _frame(self, div: str | None) -> pd.DataFrame:
        rows = {
            "Date": ["09/08/2026", "16/08/2026"],
            "HomeTeam": ["Arbroath", "Ayr"],
            "AwayTeam": ["Ayr", "Arbroath"],
            "FTHG": [1, 2],
            "FTAG": [0, 2],
            "FTR": ["H", "D"],
        }
        if div is not None:
            rows = {"Div": [div, div], **rows}
        return pd.DataFrame(rows)

    def _loader(self, data_config, tmp_path, monkeypatch, downloaded):
        """A loader whose cache is isolated and whose web fetch is stubbed."""
        import src.models.data_loader as module

        monkeypatch.setattr(module, "_CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        real_read_csv = pd.read_csv

        def fake_read_csv(source, *args, **kwargs):
            if isinstance(source, str) and source.startswith("http"):
                if isinstance(downloaded, Exception):
                    raise downloaded
                return downloaded
            return real_read_csv(source, *args, **kwargs)

        monkeypatch.setattr(module.pd, "read_csv", fake_read_csv)
        return module.FootballDataLoader(data_config)

    def test_rejects_a_payload_from_a_different_division(
        self, data_config, tmp_path, monkeypatch
    ):
        loader = self._loader(data_config, tmp_path, monkeypatch, self._frame("SC1"))
        assert loader.load_season("SP1", "2627").empty

    def test_does_not_cache_a_mismatched_payload(
        self, data_config, tmp_path, monkeypatch
    ):
        loader = self._loader(data_config, tmp_path, monkeypatch, self._frame("SC1"))
        loader.load_season("SP1", "2627")
        assert not (tmp_path / "cache" / "2627_SP1.csv").exists()

    def test_accepts_a_matching_division(self, data_config, tmp_path, monkeypatch):
        loader = self._loader(data_config, tmp_path, monkeypatch, self._frame("SP1"))
        result = loader.load_season("SP1", "2627")
        assert len(result) == 2
        assert (tmp_path / "cache" / "2627_SP1.csv").exists()

    def test_accepts_a_payload_with_no_division_column(
        self, data_config, tmp_path, monkeypatch
    ):
        """Absent evidence is not evidence of a mismatch."""
        loader = self._loader(data_config, tmp_path, monkeypatch, self._frame(None))
        assert len(loader.load_season("SP1", "2627")) == 2

    def test_rejects_an_already_poisoned_cache_file(
        self, data_config, tmp_path, monkeypatch
    ):
        """A bad file written before this check existed must not be trusted."""
        loader = self._loader(
            data_config, tmp_path, monkeypatch, RuntimeError("network down")
        )
        self._frame("SC1").to_csv(tmp_path / "cache" / "2627_SP1.csv", index=False)
        assert loader.load_season("SP1", "2627").empty

    def test_space_copy_verifies_the_division(self):
        """The Space downloads from football-data.co.uk too, so it is exposed
        to the same redirect and needs the same guard."""
        source = TestSpaceDataLoaderStaysInSync._space_loader_source()
        assert "_is_requested_division" in source
        assert 'DIVISION_COLUMN: ClassVar[str] = "Div"' in source


class TestTolerantEncoding:
    """Some football-data.co.uk season files carry a stray cp1252 byte inside
    an odds column (1819/SC0 and 1819/I2 both do). A strict UTF-8 read aborts
    on it and silently drops the entire season, so the reader falls back to
    latin-1, which decodes every byte."""

    ROWS = (
        b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H\n"
        b"SC0,04/08/2018,Celtic,Hearts,1,0,H,8\xa0\n"
        b"SC0,11/08/2018,Hearts,Celtic,2,2,D,3.5\n"
    )

    def _loader(self, data_config, tmp_path, monkeypatch):
        import src.models.data_loader as module

        cache = tmp_path / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(module, "_CACHE_DIR", cache)
        return module.FootballDataLoader(data_config), cache

    def test_reads_a_cached_file_with_cp1252_bytes(
        self, data_config, tmp_path, monkeypatch
    ):
        loader, cache = self._loader(data_config, tmp_path, monkeypatch)
        (cache / "1819_SC0.csv").write_bytes(self.ROWS)
        assert len(loader.load_season("SC0", "1819")) == 2

    def test_still_reads_plain_utf8(self, data_config, tmp_path, monkeypatch):
        loader, cache = self._loader(data_config, tmp_path, monkeypatch)
        (cache / "1819_SC0.csv").write_bytes(self.ROWS.replace(b"\xa0", b""))
        assert len(loader.load_season("SC0", "1819")) == 2


class TestMixedAndFailingDivisions:
    """Phase 3: partial failures must not take the rest of the corpus down."""

    def test_a_frame_mixing_divisions_is_rejected(self, data_config, tmp_path, monkeypatch):
        import src.models.data_loader as module

        cache = tmp_path / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(module, "_CACHE_DIR", cache)
        loader = module.FootballDataLoader(data_config)
        mixed = pd.DataFrame(
            {
                "Div": ["SP1", "SC1"],
                "Date": ["09/08/2026", "09/08/2026"],
                "HomeTeam": ["Barcelona", "Ayr"],
                "AwayTeam": ["Betis", "Arbroath"],
                "FTHG": [1, 1],
                "FTAG": [0, 0],
                "FTR": ["H", "H"],
            }
        )
        assert loader._is_requested_division(mixed, "SP1") is False

    def test_load_all_keeps_going_when_one_league_fails(
        self, data_config, tmp_path, monkeypatch
    ):
        """A 404 on one league-season must not empty the whole corpus."""
        import src.models.data_loader as module

        cache = tmp_path / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(module, "_CACHE_DIR", cache)

        good = pd.DataFrame(
            {
                "Div": ["E0"],
                "Date": ["09/08/2024"],
                "HomeTeam": ["Arsenal"],
                "AwayTeam": ["Chelsea"],
                "FTHG": [2],
                "FTAG": [0],
                "FTR": ["H"],
            }
        )

        def fake_read_csv(source, *args, **kwargs):
            if isinstance(source, str) and "E0" in source:
                return good
            raise OSError("HTTP Error 404: Not Found")

        monkeypatch.setattr(module.pd, "read_csv", fake_read_csv)
        result = module.FootballDataLoader(data_config).load_all()
        assert not result.empty
        assert set(result["League"]) == {"Premier League"}
