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

from config.config_loader import EuropeanConfig, TeamAliasConfig
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


def get_european_config(request: Request) -> EuropeanConfig:
    """UEFA settings, injected for the same reason: the country→league map
    decides which teams a queued European name may be matched against."""
    config: EuropeanConfig = request.app.state.config.european
    return config


class PendingAliasResponse(BaseModel):
    """A scraped name awaiting a decision, with advisory candidates.

    ``country`` is set for names from UEFA competitions, whose scope encodes
    it (``EU-POR``). It is what narrows the candidates to one country's
    leagues — without it, an unfamiliar full name like "AC Sparta Praha" draws
    suggestions from all 21 leagues and "Sparta Rotterdam" looks plausible.
    """

    league_code: str
    raw_name: str
    suggestions: list[str]
    country: str | None = None
    #: Every team this entry may be mapped to. Sent per entry because a UEFA
    #: scope is not a league, so the client cannot look it up in ``teams``
    #: — and should not have to know that scopes exist.
    options: list[str] = []


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


def _pending_response(
    scope: str,
    raw_name: str,
    resolver: TeamNameResolver,
    european: EuropeanConfig,
    teams: dict[str, list[str]],
) -> PendingAliasResponse:
    """One queued name with candidates scoped as narrowly as possible.

    A domestic name is scoped by its own league, exactly as before. A name
    from a UEFA competition carries its country in the scope instead, and the
    candidates come from that country's leagues — which is what stops a
    confident-looking suggestion from the wrong country.
    """
    country = _country_in_scope(scope, european)
    candidates = tuple(european.country_leagues.get(country or "", []))
    suggestions = resolver.resolve(
        TeamNameQuery(
            league_code=scope,
            raw_name=raw_name,
            candidate_league_codes=candidates,
        )
    ).suggestions
    return PendingAliasResponse(
        league_code=scope,
        raw_name=raw_name,
        suggestions=suggestions,
        country=country,
        options=sorted(_approvable_teams(scope, teams, european)),
    )


def _approvable_teams(
    scope: str, teams: dict[str, list[str]], european: EuropeanConfig
) -> list[str]:
    """Teams an alias under this scope may legitimately point at.

    A domestic scope is a league, so its own teams are the answer. A UEFA
    scope names a country instead — there is no "EU-POR" league — so the
    answer is every team in that country's leagues. Without this a European
    approval is rejected as an unknown league, which would make the whole
    review queue unusable.
    """
    if scope in teams:
        return teams[scope]
    country = _country_in_scope(scope, european)
    if country is None:
        return []
    approvable: list[str] = []
    for league in european.country_leagues.get(country, []):
        approvable.extend(teams.get(league, []))
    return approvable


def _country_in_scope(scope: str, european: EuropeanConfig) -> str | None:
    """The country a UEFA alias scope encodes, e.g. ``EU-POR`` -> ``POR``."""
    prefix = f"{european.alias_scope}-"
    if not scope.startswith(prefix):
        return None
    return scope[len(prefix) :] or None


@router.get("", response_model=TeamAliasListResponse)
async def list_team_aliases(
    _: Annotated[str, Depends(get_admin_user)],
    service: Annotated[TeamAliasService, Depends(get_alias_service)],
    team_repo: Annotated[TeamRepository, Depends(get_team_repository)],
    config: Annotated[TeamAliasConfig, Depends(get_alias_config)],
    european: Annotated[EuropeanConfig, Depends(get_european_config)],
) -> TeamAliasListResponse:
    """Approved mappings, plus the review queue with suggested matches."""
    teams = await team_repo.get_teams()
    resolver = _resolver(teams, config, service)
    pending = [
        _pending_response(
            str(row.get("league_code", "")),
            str(row.get("raw_name", "")),
            resolver,
            european,
            teams,
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
    european: Annotated[EuropeanConfig, Depends(get_european_config)],
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
    league_teams = _approvable_teams(body.league_code, teams, european)
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
