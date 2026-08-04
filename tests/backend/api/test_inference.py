"""Tests for /api/predictions/infer endpoint."""

import builtins
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api import inference as inference_module
from src.backend.api.inference import router, get_inference_service
from src.backend.core.auth import get_current_user


def _make_client(mock_service: object, user_id: str = "user-123") -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: user_id
    app.dependency_overrides[get_inference_service] = lambda: mock_service
    return TestClient(app, raise_server_exceptions=False)


class TestInferenceEndpoint:

    def test_post_returns_200_with_results(self):
        mock_svc = MagicMock()
        mock_svc.run.return_value = [
            {"home_team": "Arsenal", "away_team": "Chelsea",
             "predicted_outcome": "Home Win", "confidence": 0.65}
        ]
        res = _make_client(mock_svc).post(
            "/api/predictions/infer",
            params={"date": "28/04/2024", "league_code": "E0"},
        )
        assert res.status_code == 200

    def test_post_returns_list_in_predictions_key(self):
        mock_svc = MagicMock()
        mock_svc.run.return_value = [
            {"home_team": "Arsenal", "away_team": "Chelsea"}
        ]
        res = _make_client(mock_svc).post(
            "/api/predictions/infer",
            params={"date": "28/04/2024"},
        )
        assert "predictions" in res.json()
        assert isinstance(res.json()["predictions"], list)

    def test_post_empty_fixtures_returns_empty_list(self):
        mock_svc = MagicMock()
        mock_svc.run.return_value = []
        res = _make_client(mock_svc).post(
            "/api/predictions/infer",
            params={"date": "28/04/2024"},
        )
        assert res.json()["predictions"] == []

    def test_post_disabled_returns_503(self):
        mock_svc = MagicMock()
        mock_svc.run.side_effect = RuntimeError("inference disabled")
        res = _make_client(mock_svc).post(
            "/api/predictions/infer",
            params={"date": "28/04/2024"},
        )
        assert res.status_code == 503

    def test_unauthenticated_returns_401(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        res = client.post("/api/predictions/infer", params={"date": "28/04/2024"})
        assert res.status_code == 401

    def test_passes_league_code_to_service(self):
        mock_svc = MagicMock()
        mock_svc.run.return_value = []
        _make_client(mock_svc).post(
            "/api/predictions/infer",
            params={"date": "28/04/2024", "league_code": "SP1"},
        )
        call_kwargs = mock_svc.run.call_args
        assert "SP1" in str(call_kwargs)


_CSV_HEADER = "HomeTeam,AwayTeam\n"


def _write_cache_csv(cache_dir, league: str, rows: str) -> None:
    """Create a season CSV in the layout _load_teams_from_cache globs for."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"2024_{league}.csv").write_text(rows)


@pytest.fixture
def no_pandas(monkeypatch):
    """Simulate the deployed image, where pandas is not installed at all."""
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError("No module named 'pandas'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)


class TestLoadTeamsFromCache:
    """The CSV fallback must degrade to {} without pandas installed."""

    def test_missing_cache_dir_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(inference_module, "_CACHE_DIR", tmp_path / "absent")
        assert inference_module._load_teams_from_cache() == {}

    def test_empty_cache_dir_returns_empty(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(inference_module, "_CACHE_DIR", cache_dir)
        assert inference_module._load_teams_from_cache() == {}

    def test_no_cache_files_never_imports_pandas(
        self, tmp_path, monkeypatch, no_pandas
    ):
        """Production path: no datasets/ in the image, so pandas is never touched."""
        monkeypatch.setattr(inference_module, "_CACHE_DIR", tmp_path / "absent")
        assert inference_module._load_teams_from_cache() == {}

    def test_cache_file_without_pandas_degrades_to_empty(
        self, tmp_path, monkeypatch, no_pandas
    ):
        """Even with a cache file present, a missing pandas must not raise."""
        cache_dir = tmp_path / "cache"
        _write_cache_csv(cache_dir, "E0", _CSV_HEADER + "Arsenal,Chelsea\n")
        monkeypatch.setattr(inference_module, "_CACHE_DIR", cache_dir)
        assert inference_module._load_teams_from_cache() == {}

    def test_reads_teams_from_valid_csv(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        _write_cache_csv(
            cache_dir, "E0", _CSV_HEADER + "Arsenal,Chelsea\nChelsea,Fulham\n"
        )
        monkeypatch.setattr(inference_module, "_CACHE_DIR", cache_dir)
        assert inference_module._load_teams_from_cache() == {
            "E0": ["Arsenal", "Chelsea", "Fulham"]
        }

    def test_malformed_csv_is_skipped(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        _write_cache_csv(cache_dir, "E0", "Wrong,Columns\n1,2\n")
        monkeypatch.setattr(inference_module, "_CACHE_DIR", cache_dir)
        assert inference_module._load_teams_from_cache() == {}

    def test_only_leagues_with_cache_files_are_returned(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        _write_cache_csv(cache_dir, "SP1", _CSV_HEADER + "Real Madrid,Barcelona\n")
        monkeypatch.setattr(inference_module, "_CACHE_DIR", cache_dir)
        result = inference_module._load_teams_from_cache()
        assert list(result) == ["SP1"]


class TestGetTeamsEndpoint:

    def _client(self, mock_service) -> TestClient:
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: "user-123"
        app.dependency_overrides[get_inference_service] = lambda: mock_service
        return TestClient(app, raise_server_exceptions=False)

    def test_returns_service_data_when_leagues_differ(self):
        mock_svc = MagicMock()
        mock_svc.get_teams = AsyncMock(
            return_value={"E0": ["Arsenal"], "SP1": ["Barcelona"]}
        )
        res = self._client(mock_svc).get("/api/predictions/teams")
        assert res.status_code == 200
        assert res.json() == {"E0": ["Arsenal"], "SP1": ["Barcelona"]}

    def test_identical_league_data_falls_back_to_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(inference_module, "_CACHE_DIR", tmp_path / "absent")
        mock_svc = MagicMock()
        mock_svc.get_teams = AsyncMock(
            return_value={"E0": ["Arsenal"], "SP1": ["Arsenal"]}
        )
        res = self._client(mock_svc).get("/api/predictions/teams")
        assert res.status_code == 200
        assert res.json() == {}

    def test_service_failure_falls_back_to_cache(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        _write_cache_csv(cache_dir, "E0", _CSV_HEADER + "Arsenal,Chelsea\n")
        monkeypatch.setattr(inference_module, "_CACHE_DIR", cache_dir)
        mock_svc = MagicMock()
        mock_svc.get_teams = AsyncMock(side_effect=RuntimeError("space down"))
        res = self._client(mock_svc).get("/api/predictions/teams")
        assert res.status_code == 200
        assert res.json() == {"E0": ["Arsenal", "Chelsea"]}

    def test_single_league_response_is_not_treated_as_duplicate(self):
        mock_svc = MagicMock()
        mock_svc.get_teams = AsyncMock(return_value={"E0": ["Arsenal", "Chelsea"]})
        res = self._client(mock_svc).get("/api/predictions/teams")
        assert res.json() == {"E0": ["Arsenal", "Chelsea"]}

    def test_unauthenticated_returns_401(self):
        app = FastAPI()
        app.include_router(router)
        res = TestClient(app, raise_server_exceptions=False).get(
            "/api/predictions/teams"
        )
        assert res.status_code == 401
