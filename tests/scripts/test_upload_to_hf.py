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
