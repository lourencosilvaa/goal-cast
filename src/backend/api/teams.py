"""Team-name catalogue backing the custom-prediction pickers.

Deliberately mounted at ``/api/teams`` and **not** under ``/api/predictions``:
the predictions router owns a permissive ``GET /api/predictions/{league_code}``
that matches every sibling path registered after it, which previously made this
endpoint unreachable. See ``tests/backend/api/test_route_resolution.py``.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from src.backend.api.inference import get_inference_service
from src.backend.core.auth import get_current_user
from src.backend.repositories.team_repository import (
    FallbackTeamRepository,
    RemoteTeamRepository,
    StaticTeamRepository,
    TeamRepository,
)

router = APIRouter(prefix="/api", tags=["teams"])


def get_team_repository(request: Request) -> TeamRepository:
    """Space first (freshest), shipped registry second (always available)."""
    config = request.app.state.config
    return FallbackTeamRepository(
        [
            RemoteTeamRepository(get_inference_service(request)),
            StaticTeamRepository(config.teams.registry_path),
        ]
    )


@router.get("/teams")
async def get_teams(
    user_id: Annotated[str, Depends(get_current_user)],
    team_repo: Annotated[TeamRepository, Depends(get_team_repository)],
) -> dict[str, list[str]]:
    """Return available team names grouped by league code."""
    return await team_repo.get_teams()
