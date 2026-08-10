"""``GET /api/in-play`` — the stored prediction, re-priced on the live score.

A prediction written the night before is a statement about a match that had not
started. Once one has, the dashboard row can say something better: the same
model conditioned on the goals already scored and the minutes still left. That
is what this serves, one entry per match in progress that could be identified
confidently.

The response deliberately carries *both* numbers. A live 91% means little on
its own; next to the pre-match 72% it says the match has been decided since
kick-off, which is the thing worth showing.

Matches that could not be priced come back in ``unpriced`` with a reason
instead of being dropped. Three matches being played and two cards on screen
is a bug report waiting to happen unless the third explains itself — and the
usual explanation, a club spelling nobody has approved yet, is fixable from
the admin alias screen once it is visible.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from config.config_loader import TeamAliasConfig
from src.backend.api.results import (
    get_results_gateway,
    guard_results_call,
    split_leagues,
)
from src.backend.core.auth import get_approved_user
from src.backend.core.supabase_client import get_supabase_client
from src.backend.repositories.in_play_sources import (
    PredictionServiceSource,
    ResolverNameMatcher,
)
from src.backend.repositories.team_alias_repository import SupabaseTeamAliasRepository
from src.backend.services.in_play import InPlayCalculator
from src.backend.services.in_play_board import (
    InPlayBoard,
    InPlayBoardResult,
    PricedMatch,
    UnpricedMatch,
)
from src.backend.services.results_gateway import ResultsGateway
from src.backend.services.team_alias_service import TeamAliasService
from src.models.outcome_model import OutcomeProbabilities
from src.teams.registry import load_team_registry
from src.teams.resolver import (
    ChainedTeamAliasRepository,
    StaticTeamAliasRepository,
    TeamAliasRepository,
    TeamNameResolver,
)

router = APIRouter(prefix="/api/in-play", tags=["in-play"])


class ProbabilityTriple(BaseModel):
    home_win: float
    draw: float
    away_win: float


class InPlayMatchResponse(BaseModel):
    """One match in progress, priced twice."""

    league: str
    #: Canonical spelling, so a dashboard row can be joined on it directly.
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    #: Minutes played. ``minute_estimated`` says whether it was observed or
    #: derived from kick-off, because the free provider tier sends no clock.
    elapsed_minutes: int
    minute_estimated: bool
    status: str
    pre_match: ProbabilityTriple
    live: ProbabilityTriple
    #: Regulation minutes still to play. Minutes rather than the model's
    #: fraction: nobody reads 0.33 of a match.
    remaining_minutes: int
    #: Goals scored plus goals still expected, per side.
    expected_home_goals: float
    expected_away_goals: float


class UnpricedMatchResponse(BaseModel):
    """A match being played that has no in-play number, and why.

    The names are the provider's, not ours — when the reason is
    ``unknown_team`` that spelling is the entire diagnosis.
    """

    league: str
    home_team: str
    away_team: str
    #: ``unknown_team``, ``no_prediction`` or ``no_expected_goals``.
    reason: str


class InPlayResponse(BaseModel):
    fetched_at: str
    #: True when the live board is the results service's last good snapshot
    #: rather than a fresh one. A re-priced stale score is still stale.
    stale: bool = False
    matches: list[InPlayMatchResponse] = []
    unpriced: list[UnpricedMatchResponse] = []


def get_in_play_board(
    request: Request,
    gateway: Annotated[ResultsGateway, Depends(get_results_gateway)],
) -> InPlayBoard:
    """The board, wired to the app's own prediction and name services.

    Built per request for the same reason the gateway is: it holds no state
    worth keeping, and both of its sources cache on their own.
    """
    config = request.app.state.config
    alias_config = config.teams.aliases
    return InPlayBoard(
        gateway=gateway,
        predictions=PredictionServiceSource(request.app.state.prediction_service),
        names=ResolverNameMatcher(
            TeamNameResolver(
                canonical_teams=load_team_registry(config.teams.registry_path),
                alias_repository=_alias_repository(alias_config),
                config=alias_config,
            )
        ),
        calculator=InPlayCalculator(),
    )


def _alias_repository(alias_config: TeamAliasConfig) -> ChainedTeamAliasRepository:
    """The committed seed, plus admin-approved aliases when reachable.

    The Supabase half is built inside a ``try`` because *constructing* the
    client is what raises when the deployment has no credentials, and losing
    the runtime aliases should cost a few unresolved names — not the whole
    endpoint. Failures at read time are already absorbed by the chain.
    """
    sources: list[TeamAliasRepository] = [
        StaticTeamAliasRepository(alias_config.seed_path)
    ]
    try:
        sources.append(
            SupabaseTeamAliasRepository(
                TeamAliasService(supabase=get_supabase_client())
            )
        )
    except Exception:
        pass
    return ChainedTeamAliasRepository(sources)


@router.get("", response_model=InPlayResponse)
def in_play(
    _: Annotated[str, Depends(get_approved_user)],
    board: Annotated[InPlayBoard, Depends(get_in_play_board)],
    leagues: str = Query(
        default="", description="Comma-separated league codes; omit for all."
    ),
) -> InPlayResponse:
    return _response(guard_results_call(lambda: board.build(split_leagues(leagues))))


def _response(result: InPlayBoardResult) -> InPlayResponse:
    return InPlayResponse(
        fetched_at=str(result.fetched_at),
        stale=result.stale,
        matches=[_match(priced) for priced in result.matches],
        unpriced=[_unpriced(skipped) for skipped in result.unpriced],
    )


def _match(priced: PricedMatch) -> InPlayMatchResponse:
    forecast = priced.forecast
    return InPlayMatchResponse(
        league=priced.league,
        home_team=priced.home_team,
        away_team=priced.away_team,
        home_goals=priced.home_goals,
        away_goals=priced.away_goals,
        elapsed_minutes=priced.elapsed_minutes,
        minute_estimated=priced.minute_estimated,
        status=priced.status,
        pre_match=_triple(priced.pre_match),
        live=_triple(forecast.outcome),
        remaining_minutes=round(
            forecast.remaining_fraction * InPlayCalculator.FULL_TIME
        ),
        expected_home_goals=forecast.expected_home_goals,
        expected_away_goals=forecast.expected_away_goals,
    )


def _unpriced(skipped: UnpricedMatch) -> UnpricedMatchResponse:
    return UnpricedMatchResponse(
        league=skipped.league,
        home_team=skipped.home_team,
        away_team=skipped.away_team,
        reason=skipped.reason,
    )


def _triple(probabilities: OutcomeProbabilities) -> ProbabilityTriple:
    return ProbabilityTriple(
        home_win=probabilities.home_win,
        draw=probabilities.draw,
        away_win=probabilities.away_win,
    )
