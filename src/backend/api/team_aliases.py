"""Admin routes for validating scraped team names.

The human half of canonical-name resolution. The pipeline queues names it did
not recognise; an admin reviews each one against candidates proposed by the
resolver and either confirms a mapping or leaves it unresolved. Nothing here
decides anything on its own — a fuzzy match is only ever a suggestion.

Every route sits behind ``get_admin_user``: deciding which real team a scraped
name refers to changes what the model is asked to predict, so it is an admin
action, not a user one.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from config.config_loader import TeamAliasConfig
from src.backend.api.admin import get_admin_user
from src.backend.api.teams import get_team_repository
from src.backend.core.supabase_client import get_supabase_client
from src.backend.repositories.team_alias_repository import SupabaseTeamAliasRepository
from src.backend.repositories.team_repository import TeamRepository
from src.backend.services.team_alias_service import TeamAliasService
from src.teams.resolver import (
    ChainedTeamAliasRepository,
    StaticTeamAliasRepository,
    TeamNameQuery,
    TeamNameResolver,
)

router = APIRouter(prefix="/api/admin/team-aliases", tags=["admin", "team-aliases"])


def get_alias_service() -> TeamAliasService:
    return TeamAliasService(supabase=get_supabase_client())


def get_alias_config(request: Request) -> TeamAliasConfig:
    """Resolution settings, injected so tests state them explicitly."""
    config: TeamAliasConfig = request.app.state.config.teams.aliases
    return config


class PendingAliasResponse(BaseModel):
    """A scraped name awaiting a decision, with advisory candidates."""

    league_code: str
    raw_name: str
    suggestions: list[str]


class ApprovedAliasResponse(BaseModel):
    league_code: str
    raw_name: str
    canonical_name: str


class TeamAliasListResponse(BaseModel):
    """Everything the admin screen needs in one round trip."""

    approved: list[ApprovedAliasResponse]
    pending: list[PendingAliasResponse]
    #: Canonical names per league, so the picker offers only real teams.
    teams: dict[str, list[str]]


class ApproveAliasRequest(BaseModel):
    league_code: str
    raw_name: str
    canonical_name: str


class RevokeAliasRequest(BaseModel):
    league_code: str
    raw_name: str


class SuccessResponse(BaseModel):
    success: bool


def _resolver(
    teams: dict[str, list[str]],
    config: TeamAliasConfig,
    service: TeamAliasService,
) -> TeamNameResolver:
    """Resolver used only to propose candidates for the review screen."""
    return TeamNameResolver(
        canonical_teams=teams,
        alias_repository=ChainedTeamAliasRepository(
            [
                StaticTeamAliasRepository(config.seed_path),
                SupabaseTeamAliasRepository(service),
            ]
        ),
        config=config,
    )


@router.get("", response_model=TeamAliasListResponse)
async def list_team_aliases(
    _: Annotated[str, Depends(get_admin_user)],
    service: Annotated[TeamAliasService, Depends(get_alias_service)],
    team_repo: Annotated[TeamRepository, Depends(get_team_repository)],
    config: Annotated[TeamAliasConfig, Depends(get_alias_config)],
) -> TeamAliasListResponse:
    """Approved mappings, plus the review queue with suggested matches."""
    teams = await team_repo.get_teams()
    resolver = _resolver(teams, config, service)
    pending = [
        PendingAliasResponse(
            league_code=str(row.get("league_code", "")),
            raw_name=str(row.get("raw_name", "")),
            suggestions=resolver.resolve(
                TeamNameQuery(
                    league_code=str(row.get("league_code", "")),
                    raw_name=str(row.get("raw_name", "")),
                )
            ).suggestions,
        )
        for row in service.list_pending()
    ]
    approved = [
        ApprovedAliasResponse(
            league_code=str(row.get("league_code", "")),
            raw_name=str(row.get("raw_name", "")),
            canonical_name=str(row.get("canonical_name") or ""),
        )
        for row in service.list_approved()
        if row.get("canonical_name")
    ]
    return TeamAliasListResponse(approved=approved, pending=pending, teams=teams)


@router.post("", response_model=SuccessResponse)
async def approve_team_alias(
    body: ApproveAliasRequest,
    admin_id: Annotated[str, Depends(get_admin_user)],
    service: Annotated[TeamAliasService, Depends(get_alias_service)],
    team_repo: Annotated[TeamRepository, Depends(get_team_repository)],
) -> SuccessResponse:
    """Confirm that a scraped name refers to a specific canonical team."""
    raw_name = body.raw_name.strip()
    if not raw_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="raw_name must not be blank.",
        )

    # An approval typo would poison every later resolution, so the target is
    # validated against the live team list rather than trusted.
    teams = await team_repo.get_teams()
    league_teams = teams.get(body.league_code)
    if not league_teams:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown league '{body.league_code}'.",
        )
    if body.canonical_name not in league_teams:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"'{body.canonical_name}' is not a team in " f"'{body.league_code}'."
            ),
        )

    service.approve(
        league_code=body.league_code,
        raw_name=raw_name,
        canonical_name=body.canonical_name,
        approved_by=admin_id,
    )
    return SuccessResponse(success=True)


@router.delete("", response_model=SuccessResponse)
def revoke_team_alias(
    body: RevokeAliasRequest,
    _: Annotated[str, Depends(get_admin_user)],
    service: Annotated[TeamAliasService, Depends(get_alias_service)],
) -> SuccessResponse:
    """Remove a mapping, returning the scraped name to unresolved."""
    service.revoke(league_code=body.league_code, raw_name=body.raw_name)
    return SuccessResponse(success=True)
