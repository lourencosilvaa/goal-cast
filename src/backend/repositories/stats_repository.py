"""Sources of team and head-to-head statistics.

The deployed backend image installs only the ``backend`` dependency group — no
pandas, no model — so it cannot compute statistics itself. The numbers come
from the HuggingFace Space, which already holds the full match history in
memory. This module is the seam: a read-only interface plus the Space-backed
implementation, so a future local or cached source can be dropped in without
the API layer noticing.

Unlike :mod:`src.backend.repositories.team_repository`, a failure here is not
silently degraded to an empty answer — a statistics page showing zeros would
be worse than an honest error. Transport problems are therefore translated
into two domain errors the API maps onto status codes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx


class StatsError(RuntimeError):
    """Base class for statistics-retrieval failures."""


class TeamNotFoundError(StatsError):
    """The requested team is unknown to the upstream data set."""


class StatsUnavailableError(StatsError):
    """Statistics could not be retrieved at all (upstream down or disabled)."""


@dataclass(frozen=True)
class TeamStatsQuery:
    """Identifies one team within one competition."""

    league_code: str
    team: str


@dataclass(frozen=True)
class MatchStatsQuery:
    """Identifies a face-off between two teams of one competition."""

    league_code: str
    home_team: str
    away_team: str


class StatsRepository(ABC):
    """Read-only source of match and team statistics."""

    @abstractmethod
    async def get_match_stats(self, query: MatchStatsQuery) -> dict[str, Any]:
        """Head-to-head history plus both team profiles for a fixture."""

    @abstractmethod
    async def get_team_stats(self, query: TeamStatsQuery) -> dict[str, Any]:
        """Full statistical profile of a single team."""


class RemoteStatsRepository(StatsRepository):
    """Statistics served by the HuggingFace Space via the InferenceService."""

    #: Upstream status meaning "no such team", as opposed to "cannot answer".
    NOT_FOUND_STATUS: ClassVar[int] = 404

    def __init__(self, inference_service: Any) -> None:
        self._inference_service = inference_service

    async def get_match_stats(self, query: MatchStatsQuery) -> dict[str, Any]:
        return await self._fetch(
            self._inference_service.get_match_insights(
                home_team=query.home_team,
                away_team=query.away_team,
                league_code=query.league_code,
            )
        )

    async def get_team_stats(self, query: TeamStatsQuery) -> dict[str, Any]:
        return await self._fetch(
            self._inference_service.get_team_insights(
                team=query.team, league_code=query.league_code
            )
        )

    async def _fetch(self, awaitable: Any) -> dict[str, Any]:
        """Await an upstream call, mapping its failures onto domain errors."""
        try:
            payload: dict[str, Any] = await awaitable
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == self.NOT_FOUND_STATUS:
                raise TeamNotFoundError(str(exc)) from exc
            raise StatsUnavailableError(str(exc)) from exc
        except Exception as exc:
            raise StatsUnavailableError(str(exc)) from exc
        return payload
