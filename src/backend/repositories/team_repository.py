"""Team-name sources behind the custom-prediction team pickers.

Three interchangeable implementations of a single read-only interface:

* :class:`RemoteTeamRepository` — the HuggingFace Space, always freshest.
* :class:`StaticTeamRepository` — a JSON registry shipped inside the image.
* :class:`FallbackTeamRepository` — chains sources, first non-empty wins.

Every implementation degrades to an empty mapping instead of raising, so a
missing file or an unreachable Space can never take the endpoint down.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar


class TeamRepository(ABC):
    """Read-only source of team names grouped by league code."""

    @abstractmethod
    async def get_teams(self) -> dict[str, list[str]]:
        """Return ``{league_code: [team, ...]}``, or ``{}`` when unavailable."""


class StaticTeamRepository(TeamRepository):
    """Teams read from a JSON registry file on disk."""

    ENCODING: ClassVar[str] = "utf-8"

    def __init__(self, registry_path: str | Path) -> None:
        self._registry_path = Path(registry_path)

    async def get_teams(self) -> dict[str, list[str]]:
        try:
            raw = json.loads(self._registry_path.read_text(encoding=self.ENCODING))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            str(code): sorted({str(team) for team in teams})
            for code, teams in raw.items()
            if isinstance(teams, list) and teams
        }


class RemoteTeamRepository(TeamRepository):
    """Teams served by the HuggingFace Space, via the shared InferenceService."""

    # Two or more leagues sharing one identical list means the upstream
    # per-league grouping collapsed; the data is unusable rather than merely
    # incomplete, so it is rejected wholesale.
    MIN_LEAGUES_FOR_DUPLICATE_CHECK: ClassVar[int] = 2

    def __init__(self, inference_service: Any) -> None:
        self._inference_service = inference_service

    async def get_teams(self) -> dict[str, list[str]]:
        try:
            data = await self._inference_service.get_teams()
        except Exception:
            return {}
        if not isinstance(data, dict) or not data:
            return {}
        team_lists = list(data.values())
        if len(team_lists) >= self.MIN_LEAGUES_FOR_DUPLICATE_CHECK and all(
            teams == team_lists[0] for teams in team_lists[1:]
        ):
            return {}
        return data


class FallbackTeamRepository(TeamRepository):
    """Queries sources in order and returns the first non-empty result."""

    def __init__(self, sources: Sequence[TeamRepository]) -> None:
        self._sources = list(sources)

    async def get_teams(self) -> dict[str, list[str]]:
        for source in self._sources:
            teams = await source.get_teams()
            if teams:
                return teams
        return {}
