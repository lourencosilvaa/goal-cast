"""Tests for the GET /api/teams endpoint.

The endpoint delegates to an injected TeamRepository — no pandas, no
``datasets/`` directory, and no silent dependence on the live HF Space.
Route placement is guarded separately in ``test_route_resolution.py``.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api import teams as teams_module
from src.backend.services.inference_service import InferenceService
from src.backend.api.teams import router, get_team_repository
from src.backend.core.auth import get_current_user


def _registry(tmp_path, payload) -> str:
    """Write an explicit static registry file and return its path."""
    path = tmp_path / "teams_registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestGetTeamsEndpoint:

    def _client(self, repository) -> TestClient:
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: "user-123"
        app.dependency_overrides[get_team_repository] = lambda: repository
        return TestClient(app, raise_server_exceptions=False)

    def test_returns_repository_data(self):
        repo = AsyncMock()
        repo.get_teams.return_value = {"E0": ["Arsenal"], "SP1": ["Barcelona"]}
        res = self._client(repo).get("/api/teams")
        assert res.status_code == 200
        assert res.json() == {"E0": ["Arsenal"], "SP1": ["Barcelona"]}

    def test_empty_repository_returns_empty_object(self):
        repo = AsyncMock()
        repo.get_teams.return_value = {}
        res = self._client(repo).get("/api/teams")
        assert res.status_code == 200
        assert res.json() == {}

    def test_unauthenticated_returns_401(self):
        app = FastAPI()
        app.include_router(router)
        res = TestClient(app, raise_server_exceptions=False).get("/api/teams")
        assert res.status_code == 401


class TestTeamRepositoryProvider:
    """get_team_repository must wire the Space first and the static registry
    second, using only values taken from the injected config."""

    @staticmethod
    def _request(registry_path: str) -> MagicMock:
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
            InferenceService, "get_teams", AsyncMock(return_value={})
        )
        repo = teams_module.get_team_repository(self._request(path))
        assert await repo.get_teams() == {"E0": ["Arsenal", "Chelsea"]}

    @pytest.mark.asyncio
    async def test_space_data_wins_when_available(self, tmp_path, monkeypatch):
        path = _registry(tmp_path, {"E0": ["Stale FC"]})
        monkeypatch.setattr(
            InferenceService,
            "get_teams",
            AsyncMock(return_value={"E0": ["Arsenal"], "SP1": ["Barcelona"]}),
        )
        repo = teams_module.get_team_repository(self._request(path))
        assert await repo.get_teams() == {"E0": ["Arsenal"], "SP1": ["Barcelona"]}

    @pytest.mark.asyncio
    async def test_space_failure_falls_back_to_registry(self, tmp_path, monkeypatch):
        path = _registry(tmp_path, {"E0": ["Arsenal"]})
        monkeypatch.setattr(
            InferenceService,
            "get_teams",
            AsyncMock(side_effect=RuntimeError("space down")),
        )
        repo = teams_module.get_team_repository(self._request(path))
        assert await repo.get_teams() == {"E0": ["Arsenal"]}


class TestShippedTeamsRegistry:
    """The registry file is the deployed fallback — it must exist and be sane."""

    @staticmethod
    def _registry_data() -> tuple[dict, dict]:
        from config.config_loader import load_config

        config = load_config("config/config.yaml")
        data = json.loads(
            Path(config.teams.registry_path).read_text(encoding="utf-8")
        )
        return config.data.leagues, data

    def test_registry_path_from_config_exists(self):
        from config.config_loader import load_config

        assert Path(load_config("config/config.yaml").teams.registry_path).exists()

    def test_registry_covers_every_configured_league(self):
        leagues, registry = self._registry_data()
        assert set(registry) == set(leagues)

    def test_each_league_has_teams(self):
        _, registry = self._registry_data()
        assert all(len(teams) >= 2 for teams in registry.values())

    def test_leagues_do_not_share_identical_team_lists(self):
        _, registry = self._registry_data()
        lists = [tuple(v) for v in registry.values()]
        assert len(set(lists)) == len(lists)
