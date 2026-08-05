"""Phase 1 tests for the team-name repositories behind GET /api/predictions/teams.

Each repository is exercised in isolation with an explicitly built source
(temporary JSON file or mocked inference service) — no reliance on the shipped
registry or on a live HuggingFace Space.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.backend.repositories.team_repository import (
    FallbackTeamRepository,
    RemoteTeamRepository,
    StaticTeamRepository,
    TeamRepository,
)


def _write_registry(path, payload) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestTeamRepositoryContract:

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            TeamRepository()  # type: ignore[abstract]

    def test_implementations_are_team_repositories(self, tmp_path):
        static = StaticTeamRepository(tmp_path / "teams.json")
        remote = RemoteTeamRepository(AsyncMock())
        assert isinstance(static, TeamRepository)
        assert isinstance(remote, TeamRepository)
        assert isinstance(FallbackTeamRepository([static, remote]), TeamRepository)


class TestStaticTeamRepository:

    @pytest.mark.asyncio
    async def test_reads_teams_from_registry_file(self, tmp_path):
        path = _write_registry(
            tmp_path / "teams.json",
            {"E0": ["Arsenal", "Chelsea"], "SP1": ["Barcelona"]},
        )
        assert await StaticTeamRepository(path).get_teams() == {
            "E0": ["Arsenal", "Chelsea"],
            "SP1": ["Barcelona"],
        }

    @pytest.mark.asyncio
    async def test_team_names_are_sorted_and_deduplicated(self, tmp_path):
        path = _write_registry(
            tmp_path / "teams.json", {"E0": ["Chelsea", "Arsenal", "Chelsea"]}
        )
        assert await StaticTeamRepository(path).get_teams() == {
            "E0": ["Arsenal", "Chelsea"]
        }

    @pytest.mark.asyncio
    async def test_missing_file_returns_empty(self, tmp_path):
        repo = StaticTeamRepository(tmp_path / "absent.json")
        assert await repo.get_teams() == {}

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty(self, tmp_path):
        path = tmp_path / "teams.json"
        path.write_text("{not json", encoding="utf-8")
        assert await StaticTeamRepository(path).get_teams() == {}

    @pytest.mark.asyncio
    async def test_non_object_json_returns_empty(self, tmp_path):
        path = tmp_path / "teams.json"
        path.write_text('["Arsenal"]', encoding="utf-8")
        assert await StaticTeamRepository(path).get_teams() == {}

    @pytest.mark.asyncio
    async def test_directory_instead_of_file_returns_empty(self, tmp_path):
        (tmp_path / "teams.json").mkdir()
        assert await StaticTeamRepository(tmp_path / "teams.json").get_teams() == {}

    @pytest.mark.asyncio
    async def test_leagues_with_empty_or_invalid_values_are_dropped(self, tmp_path):
        path = _write_registry(
            tmp_path / "teams.json",
            {"E0": ["Arsenal"], "SP1": [], "D1": "Bayern", "I1": None},
        )
        assert await StaticTeamRepository(path).get_teams() == {"E0": ["Arsenal"]}

    @pytest.mark.asyncio
    async def test_empty_registry_object_returns_empty(self, tmp_path):
        path = _write_registry(tmp_path / "teams.json", {})
        assert await StaticTeamRepository(path).get_teams() == {}

    @pytest.mark.asyncio
    async def test_accepts_path_object_and_string(self, tmp_path):
        payload = {"E0": ["Arsenal"]}
        path = _write_registry(tmp_path / "teams.json", payload)
        assert await StaticTeamRepository(path).get_teams() == payload
        assert await StaticTeamRepository(Path(path)).get_teams() == payload


class TestRemoteTeamRepository:

    @pytest.mark.asyncio
    async def test_returns_service_data(self):
        svc = AsyncMock()
        svc.get_teams.return_value = {"E0": ["Arsenal"], "SP1": ["Barcelona"]}
        assert await RemoteTeamRepository(svc).get_teams() == {
            "E0": ["Arsenal"],
            "SP1": ["Barcelona"],
        }

    @pytest.mark.asyncio
    async def test_empty_service_response_returns_empty(self):
        """The live Space regression: /teams answered {} and the UI went blank."""
        svc = AsyncMock()
        svc.get_teams.return_value = {}
        assert await RemoteTeamRepository(svc).get_teams() == {}

    @pytest.mark.asyncio
    async def test_identical_league_lists_are_rejected(self):
        svc = AsyncMock()
        svc.get_teams.return_value = {"E0": ["Arsenal"], "SP1": ["Arsenal"]}
        assert await RemoteTeamRepository(svc).get_teams() == {}

    @pytest.mark.asyncio
    async def test_service_failure_returns_empty(self):
        svc = AsyncMock()
        svc.get_teams.side_effect = RuntimeError("space down")
        assert await RemoteTeamRepository(svc).get_teams() == {}

    @pytest.mark.asyncio
    async def test_network_error_returns_empty(self):
        import httpx

        svc = AsyncMock()
        svc.get_teams.side_effect = httpx.ConnectError("connection refused")
        assert await RemoteTeamRepository(svc).get_teams() == {}

    @pytest.mark.asyncio
    async def test_non_dict_response_returns_empty(self):
        svc = AsyncMock()
        svc.get_teams.return_value = ["Arsenal", "Chelsea"]
        assert await RemoteTeamRepository(svc).get_teams() == {}

    @pytest.mark.asyncio
    async def test_single_league_is_not_treated_as_duplicate(self):
        svc = AsyncMock()
        svc.get_teams.return_value = {"E0": ["Arsenal", "Chelsea"]}
        assert await RemoteTeamRepository(svc).get_teams() == {
            "E0": ["Arsenal", "Chelsea"]
        }

    @pytest.mark.asyncio
    async def test_partial_overlap_between_leagues_is_kept(self):
        svc = AsyncMock()
        svc.get_teams.return_value = {"E0": ["Arsenal"], "SP1": ["Arsenal", "Sevilla"]}
        assert await RemoteTeamRepository(svc).get_teams() == {
            "E0": ["Arsenal"],
            "SP1": ["Arsenal", "Sevilla"],
        }


class TestFallbackTeamRepository:

    @pytest.mark.asyncio
    async def test_first_non_empty_source_wins(self, tmp_path):
        primary = AsyncMock()
        primary.get_teams.return_value = {"E0": ["Arsenal"]}
        secondary = StaticTeamRepository(
            _write_registry(tmp_path / "teams.json", {"E0": ["Fulham"]})
        )
        repo = FallbackTeamRepository([primary, secondary])
        assert await repo.get_teams() == {"E0": ["Arsenal"]}

    @pytest.mark.asyncio
    async def test_empty_primary_falls_through_to_secondary(self, tmp_path):
        primary = AsyncMock()
        primary.get_teams.return_value = {}
        secondary = StaticTeamRepository(
            _write_registry(tmp_path / "teams.json", {"E0": ["Fulham"]})
        )
        repo = FallbackTeamRepository([primary, secondary])
        assert await repo.get_teams() == {"E0": ["Fulham"]}

    @pytest.mark.asyncio
    async def test_all_sources_empty_returns_empty(self):
        first, second = AsyncMock(), AsyncMock()
        first.get_teams.return_value = {}
        second.get_teams.return_value = {}
        assert await FallbackTeamRepository([first, second]).get_teams() == {}

    @pytest.mark.asyncio
    async def test_later_sources_are_not_queried_after_a_hit(self):
        first, second = AsyncMock(), AsyncMock()
        first.get_teams.return_value = {"E0": ["Arsenal"]}
        second.get_teams.return_value = {"E0": ["Fulham"]}
        await FallbackTeamRepository([first, second]).get_teams()
        second.get_teams.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_sources_returns_empty(self):
        assert await FallbackTeamRepository([]).get_teams() == {}

    @pytest.mark.asyncio
    async def test_real_chain_survives_a_broken_space_and_missing_file(self):
        """Worst case: Space down and registry absent — degrade, never raise."""
        svc = AsyncMock()
        svc.get_teams.side_effect = RuntimeError("space down")
        repo = FallbackTeamRepository(
            [RemoteTeamRepository(svc), StaticTeamRepository("/nonexistent/teams.json")]
        )
        assert await repo.get_teams() == {}

    @pytest.mark.asyncio
    async def test_source_order_is_preserved(self):
        """A source list is not reordered — construction order defines priority."""
        first, second, third = AsyncMock(), AsyncMock(), AsyncMock()
        first.get_teams.return_value = {}
        second.get_teams.return_value = {}
        third.get_teams.return_value = {"E0": ["Arsenal"]}
        assert await FallbackTeamRepository([first, second, third]).get_teams() == {
            "E0": ["Arsenal"]
        }
        first.get_teams.assert_awaited_once()
        second.get_teams.assert_awaited_once()
