"""Phase 1 tests for the Hugging Face upload script.

The dataset snapshot and the model artefacts are published on independent
schedules: a held-back refit must still be able to ship fresh data. These
tests pin the `--datasets-only` mode that makes that possible, using a fake
Hub API so nothing leaves the machine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.upload_to_hf import _build_parquet_datasets, main  # noqa: E402

_HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
_ROWS = (
    "E0,10/08/2025,Arsenal,Chelsea,2,0,H\n"
    "E0,17/08/2025,Chelsea,Arsenal,1,1,D\n"
)
_REPO = "tester/football-model"
_TOKEN = "hf_test_token"


class FakeHfApi:
    """Records Hub calls in place of ``huggingface_hub.HfApi``."""

    def __init__(self) -> None:
        self.created: list[str] = []
        self.uploads: list[dict] = []

    def create_repo(self, **kwargs) -> None:
        self.created.append(kwargs.get("repo_id", ""))

    def upload_folder(self, **kwargs) -> None:
        self.uploads.append(kwargs)


def _cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "2526_E0.csv").write_text(_HEADER + _ROWS, encoding="utf-8")
    return cache


def _models(tmp_path: Path, *, with_artefacts: bool) -> Path:
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    if with_artefacts:
        (models / "ensemble_model.joblib").write_bytes(b"fake-model")
        (models / "training_results.json").write_text("{}", encoding="utf-8")
    return models


def _argv(models: Path, cache: Path, *extra: str) -> list[str]:
    return [
        "upload_to_hf.py",
        "--models-dir",
        str(models),
        "--cache-dir",
        str(cache),
        "--repo-id",
        _REPO,
        "--token",
        _TOKEN,
        *extra,
    ]


def _paths_in_repo(api: FakeHfApi) -> list[str]:
    return [upload.get("path_in_repo", "") for upload in api.uploads]


class TestBuildParquetDatasets:

    def test_writes_one_parquet_per_league(self, tmp_path):
        result = _build_parquet_datasets(_cache(tmp_path), tmp_path / "out")
        assert set(result) == {"E0"}
        assert result["E0"].exists()

    def test_returns_nothing_when_cache_is_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _build_parquet_datasets(empty, tmp_path / "out") == {}


class TestDatasetsOnlyMode:

    def test_uploads_datasets_and_no_artefacts(self, tmp_path, monkeypatch):
        api = FakeHfApi()
        monkeypatch.setattr(
            sys,
            "argv",
            _argv(_models(tmp_path, with_artefacts=True), _cache(tmp_path), "--datasets-only"),
        )
        main(api=api)
        assert _paths_in_repo(api) == ["datasets"]

    def test_works_when_no_model_artefacts_exist(self, tmp_path, monkeypatch):
        """A data-only run must not require a model that was never rebuilt."""
        api = FakeHfApi()
        monkeypatch.setattr(
            sys,
            "argv",
            _argv(_models(tmp_path, with_artefacts=False), _cache(tmp_path), "--datasets-only"),
        )
        main(api=api)
        assert _paths_in_repo(api) == ["datasets"]


class TestFullMode:

    def test_uploads_both_artefacts_and_datasets(self, tmp_path, monkeypatch):
        api = FakeHfApi()
        monkeypatch.setattr(
            sys,
            "argv",
            _argv(_models(tmp_path, with_artefacts=True), _cache(tmp_path)),
        )
        main(api=api)
        assert "datasets" in _paths_in_repo(api)
        assert len(api.uploads) == 2

    def test_dry_run_uploads_nothing(self, tmp_path, monkeypatch):
        api = FakeHfApi()
        monkeypatch.setattr(
            sys,
            "argv",
            _argv(_models(tmp_path, with_artefacts=True), _cache(tmp_path), "--dry-run"),
        )
        main(api=api)
        assert api.uploads == []


class TestMixedDtypeColumns:
    """Old football-data CSVs carry columns whose type varies by season.

    ``BbAH`` reads as int64 in one season and object in another once a stray
    non-numeric value appears, so concatenating seasons yields a column
    holding both — and pyarrow refuses to write it:

        ArrowInvalid: Could not convert '17' with type str

    That aborted the whole upload, including the model artefacts. It cannot
    be left to chance: every one of these columns is discarded by the loader
    on read anyway, since it applies ``columns_to_keep`` immediately after
    ``read_parquet``.
    """

    def _cache_with_mixed_types(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "2425_I2.csv").write_text(
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,BbAH\n"
            "I2,10/08/2024,Bari,Como,2,0,H,17\n",
            encoding="utf-8",
        )
        (cache / "2526_I2.csv").write_text(
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,BbAH\n"
            "I2,10/08/2025,Como,Bari,1,1,D,x\n",
            encoding="utf-8",
        )
        return cache

    def test_mixed_types_no_longer_abort_the_upload(self, tmp_path):
        result = _build_parquet_datasets(
            self._cache_with_mixed_types(tmp_path), tmp_path / "out"
        )
        assert result["I2"].exists()

    def test_the_written_file_is_readable(self, tmp_path):
        import pandas as pd

        result = _build_parquet_datasets(
            self._cache_with_mixed_types(tmp_path), tmp_path / "out"
        )
        assert len(pd.read_parquet(result["I2"])) == 2

    def test_columns_the_loader_discards_are_not_uploaded(self, tmp_path):
        import pandas as pd

        result = _build_parquet_datasets(
            self._cache_with_mixed_types(tmp_path), tmp_path / "out"
        )
        assert "BbAH" not in pd.read_parquet(result["I2"]).columns

    def test_every_column_the_loader_keeps_survives(self, tmp_path):
        import pandas as pd

        result = _build_parquet_datasets(_cache(tmp_path), tmp_path / "out")
        columns = set(pd.read_parquet(result["E0"]).columns)
        for kept in ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"):
            assert kept in columns

    def test_season_is_preserved(self, tmp_path):
        """The loader filters by season, so this column must survive."""
        import pandas as pd

        result = _build_parquet_datasets(_cache(tmp_path), tmp_path / "out")
        assert "Season" in pd.read_parquet(result["E0"]).columns

    def test_numeric_columns_stay_numeric(self, tmp_path):
        import pandas as pd

        result = _build_parquet_datasets(_cache(tmp_path), tmp_path / "out")
        frame = pd.read_parquet(result["E0"])
        assert frame["FTHG"].dtype.kind in "iuf"


class TestEuropeanDataset:
    """The Space reads cross-league history from this file.

    Uploaded already translated to canonical keys, so the Space needs neither
    Supabase nor the alias registry to reproduce the model's ELO features.
    """

    def test_is_written_alongside_the_league_datasets(self, tmp_path):
        # Patched at its source: the import inside _build_european_dataset is
        # function-local, so the module attribute is never consulted.
        from unittest.mock import patch

        import pandas as pd

        from scripts import upload_to_hf

        corpus = pd.DataFrame(
            {"Div": ["CL"], "HomeTeam": ["Benfica"], "AwayTeam": ["Arsenal"]}
        )
        datasets = tmp_path / "datasets"
        datasets.mkdir(parents=True)
        with patch(
            "src.corpus.european_corpus.load_european_corpus", return_value=corpus
        ):
            path = upload_to_hf._build_european_dataset(object(), datasets)
        assert path is not None and path.name == upload_to_hf.EUROPEAN_DATASET
        assert len(pd.read_parquet(path)) == 1

    def test_absent_corpus_writes_nothing(self, tmp_path):
        from unittest.mock import patch

        import pandas as pd

        from scripts import upload_to_hf

        datasets = tmp_path / "datasets"
        datasets.mkdir(parents=True)
        with patch(
            "src.corpus.european_corpus.load_european_corpus",
            return_value=pd.DataFrame(),
        ):
            assert upload_to_hf._build_european_dataset(object(), datasets) is None
