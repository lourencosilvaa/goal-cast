"""Team and head-to-head statistics for the prediction and team pages.

Mounted at ``/api/stats`` and deliberately **not** under ``/api/predictions``:
the predictions router owns a permissive ``GET /api/predictions/{league_code}``
that swallows every sibling GET path registered after it. See
``tests/backend/api/test_route_resolution.py``.

The router holds no statistics logic. It resolves a repository, translates the
request into a query object and validates the answer against an explicit
response contract, so the frontend has a typed shape to code against even
though the numbers are produced upstream.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from src.backend.api.inference import get_inference_service
from src.backend.core.auth import get_current_user
from src.backend.repositories.stats_repository import (
    MatchStatsQuery,
    RemoteStatsRepository,
    StatsRepository,
    StatsUnavailableError,
    TeamNotFoundError,
    TeamStatsQuery,
)

router = APIRouter(prefix="/api/stats", tags=["stats"])


def get_stats_repository(request: Request) -> StatsRepository:
    """Statistics come from the Space, which owns the historical data."""
    return RemoteStatsRepository(get_inference_service(request))


class TeamRecordResponse(BaseModel):
    """Win/draw/loss and goal counters plus their derived ratios."""

    played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    points: int
    points_per_game: float
    win_pct: float
    draw_pct: float
    loss_pct: float
    avg_goals_for: float
    avg_goals_against: float
    goal_difference: int


class MatchRecordResponse(BaseModel):
    """One past match; ``result``/``venue`` are relative to the queried team."""

    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    result: str | None = None
    venue: str | None = None


class TeamRatesResponse(BaseModel):
    clean_sheets: float
    failed_to_score: float
    btts: float
    over_2_5: float


class TeamAveragesResponse(BaseModel):
    shots: float
    shots_on_target: float
    corners: float
    cards: float


class TeamStatsResponse(BaseModel):
    """Statistical profile of one team within one competition."""

    team: str
    league: str
    league_code: str
    overall: TeamRecordResponse
    home: TeamRecordResponse
    away: TeamRecordResponse
    recent: TeamRecordResponse
    form_sequence: list[str]
    rates: TeamRatesResponse
    averages: TeamAveragesResponse
    recent_matches: list[MatchRecordResponse]


class HeadToHeadResponse(BaseModel):
    """Past meetings, counted from the upcoming fixture's point of view."""

    home_team: str
    away_team: str
    played: int
    home_wins: int
    draws: int
    away_wins: int
    home_goals: int
    away_goals: int
    avg_goals_home: float
    avg_goals_away: float
    avg_goals_total: float
    btts_pct: float
    over_2_5_pct: float
    matches: list[MatchRecordResponse]


class ExpectedGoalsResponse(BaseModel):
    home: float
    away: float
    total: float


class OverUnderResponse(BaseModel):
    over_1_5: float
    over_2_5: float
    over_3_5: float
    under_2_5: float


class BttsResponse(BaseModel):
    yes: float
    no: float


class ScorelineResponse(BaseModel):
    score: str
    prob: float


class GoalMarketsResponse(BaseModel):
    """Score-model markets; absent when the model does not know both teams."""

    expected_goals: ExpectedGoalsResponse
    over_under: OverUnderResponse
    btts: BttsResponse
    top_scorelines: list[ScorelineResponse]


class MatchStatsResponse(BaseModel):
    """Everything known about a face-off between two teams."""

    home_team: str
    away_team: str
    league: str
    league_code: str
    head_to_head: HeadToHeadResponse
    home: TeamStatsResponse
    away: TeamStatsResponse
    goal_markets: GoalMarketsResponse | None = None


class MatchStatsRequest(BaseModel):
    home_team: str
    away_team: str
    league_code: str


def _http_error(exc: Exception, not_found: bool) -> HTTPException:
    if not_found:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
    )


@router.post("/match", response_model=MatchStatsResponse)
async def get_match_stats(
    body: MatchStatsRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    stats_repo: Annotated[StatsRepository, Depends(get_stats_repository)],
) -> MatchStatsResponse:
    """Head-to-head record, both team profiles and the goal markets."""
    query = MatchStatsQuery(
        league_code=body.league_code,
        home_team=body.home_team,
        away_team=body.away_team,
    )
    try:
        payload = await stats_repo.get_match_stats(query)
    except TeamNotFoundError as exc:
        raise _http_error(exc, not_found=True)
    except StatsUnavailableError as exc:
        raise _http_error(exc, not_found=False)
    return MatchStatsResponse(**payload)


@router.get("/team", response_model=TeamStatsResponse)
async def get_team_stats(
    user_id: Annotated[str, Depends(get_current_user)],
    stats_repo: Annotated[StatsRepository, Depends(get_stats_repository)],
    league_code: Annotated[str, Query()],
    team: Annotated[str, Query()],
) -> TeamStatsResponse:
    """Full statistical profile of a single team."""
    query = TeamStatsQuery(league_code=league_code, team=team)
    try:
        payload = await stats_repo.get_team_stats(query)
    except TeamNotFoundError as exc:
        raise _http_error(exc, not_found=True)
    except StatsUnavailableError as exc:
        raise _http_error(exc, not_found=False)
    return TeamStatsResponse(**payload)
