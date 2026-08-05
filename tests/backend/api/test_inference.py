"""Tests for the /api/predictions/infer and /api/predictions/teams endpoints."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api import inference as inference_module
from src.backend.api.inference import (
    router,
    get_inference_service,
    get_team_repository,
)
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
        mock_svc.run = AsyncMock()
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
        mock_svc.run = AsyncMock()
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
        mock_svc.run = AsyncMock()
        mock_svc.run.return_value = []
        res = _make_client(mock_svc).post(
            "/api/predictions/infer",
            params={"date": "28/04/2024"},
        )
        assert res.json()["predictions"] == []

    def test_post_disabled_returns_503(self):
        mock_svc = MagicMock()
        mock_svc.run = AsyncMock()
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
        mock_svc.run = AsyncMock()
        mock_svc.run.return_value = []
        _make_client(mock_svc).post(
            "/api/predictions/infer",
            params={"date": "28/04/2024", "league_code": "SP1"},
        )
        call_kwargs = mock_svc.run.call_args
        assert "SP1" in str(call_kwargs)


def _registry(tmp_path, payload) -> str:
    """Write an explicit static registry file and return its path."""
    path = tmp_path / "teams_registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestGetTeamsEndpoint:
    """The endpoint delegates to an injected TeamRepository — no pandas, no
    datasets/ directory, and no silent dependence on the live HF Space."""

    def _client(self, repository) -> TestClient:
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: "user-123"
        app.dependency_overrides[get_team_repository] = lambda: repository
        return TestClient(app, raise_server_exceptions=False)

    def test_returns_repository_data(self):
        repo = AsyncMock()
        repo.get_teams.return_value = {"E0": ["Arsenal"], "SP1": ["Barcelona"]}
        res = self._client(repo).get("/api/predictions/teams")
        assert res.status_code == 200
        assert res.json() == {"E0": ["Arsenal"], "SP1": ["Barcelona"]}

    def test_empty_repository_returns_empty_object(self):
        repo = AsyncMock()
        repo.get_teams.return_value = {}
        res = self._client(repo).get("/api/predictions/teams")
        assert res.status_code == 200
        assert res.json() == {}

    def test_unauthenticated_returns_401(self):
        app = FastAPI()
        app.include_router(router)
        res = TestClient(app, raise_server_exceptions=False).get(
            "/api/predictions/teams"
        )
        assert res.status_code == 401


class TestTeamRepositoryProvider:
    """get_team_repository must wire the Space first and the static registry
    second, using only values taken from the injected config."""

    @staticmethod
    def _request(registry_path: str, space_teams) -> MagicMock:
        config = MagicMock()
        config.inference.enabled = True
        config.inference.space_url = "https://space.test"
        config.teams.registry_path = registry_path
        request = MagicMock()
        request.app.state.config = config
        return request

    @pytest.mark.asyncio
    async def test_falls_back_to_registry_when_space_returns_empty(
        self, tmp_path, monkeypatch
    ):
        """The live regression: the Space answered {} and the UI went blank."""
        path = _registry(tmp_path, {"E0": ["Arsenal", "Chelsea"]})
        monkeypatch.setattr(
            inference_module.InferenceService,
            "get_teams",
            AsyncMock(return_value={}),
        )
        repo = inference_module.get_team_repository(self._request(path, {}))
        assert await repo.get_teams() == {"E0": ["Arsenal", "Chelsea"]}

    @pytest.mark.asyncio
    async def test_space_data_wins_when_available(self, tmp_path, monkeypatch):
        path = _registry(tmp_path, {"E0": ["Stale FC"]})
        monkeypatch.setattr(
            inference_module.InferenceService,
            "get_teams",
            AsyncMock(return_value={"E0": ["Arsenal"], "SP1": ["Barcelona"]}),
        )
        repo = inference_module.get_team_repository(self._request(path, {}))
        assert await repo.get_teams() == {"E0": ["Arsenal"], "SP1": ["Barcelona"]}

    @pytest.mark.asyncio
    async def test_space_failure_falls_back_to_registry(self, tmp_path, monkeypatch):
        path = _registry(tmp_path, {"E0": ["Arsenal"]})
        monkeypatch.setattr(
            inference_module.InferenceService,
            "get_teams",
            AsyncMock(side_effect=RuntimeError("space down")),
        )
        repo = inference_module.get_team_repository(self._request(path, {}))
        assert await repo.get_teams() == {"E0": ["Arsenal"]}


class TestShippedTeamsRegistry:
    """The registry file is the deployed fallback — it must exist and be sane."""

    def test_registry_path_from_config_exists(self):
        from config.config_loader import load_config

        path = Path(load_config("config/config.yaml").teams.registry_path)
        assert path.exists()

    def test_registry_covers_every_configured_league(self):
        from config.config_loader import load_config

        config = load_config("config/config.yaml")
        registry = json.loads(
            Path(config.teams.registry_path).read_text(encoding="utf-8")
        )
        assert set(registry) == set(config.data.leagues)

    def test_each_league_has_teams(self):
        from config.config_loader import load_config

        config = load_config("config/config.yaml")
        registry = json.loads(
            Path(config.teams.registry_path).read_text(encoding="utf-8")
        )
        assert all(len(teams) >= 2 for teams in registry.values())

    def test_leagues_do_not_share_identical_team_lists(self):
        from config.config_loader import load_config

        config = load_config("config/config.yaml")
        registry = json.loads(
            Path(config.teams.registry_path).read_text(encoding="utf-8")
        )
        lists = [tuple(v) for v in registry.values()]
        assert len(set(lists)) == len(lists)
