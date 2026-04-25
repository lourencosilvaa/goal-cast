"""Predictions endpoint — returns ML predictions grouped by league."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from src.backend.core.auth import get_approved_user

router = APIRouter(prefix="/api", tags=["predictions"])


class ProbabilitiesResponse(BaseModel):
    home_win: float
    draw: float
    away_win: float


class OddsResponse(BaseModel):
    home: float
    draw: float
    away: float


class ImpliedProbsResponse(BaseModel):
    home: float
    draw: float
    away: float


class OverUnderResponse(BaseModel):
    over_15: float
    over_25: float
    over_35: float
    under_25: float


class BttsResponse(BaseModel):
    yes: float
    no: float


class ScorelineResponse(BaseModel):
    score: str
    prob: float


class FormResponse(BaseModel):
    home: float
    away: float


class ExpectedGoalsResponse(BaseModel):
    home: float
    away: float
    total: float


class ValueBetResponse(BaseModel):
    outcome: str
    ml_probability: float
    bookmaker_implied: float
    edge: float
    edge_pct: str
    best_odds: float
    kelly_fraction: float
    confidence: str


class MatchPredictionResponse(BaseModel):
    home_team: str
    away_team: str
    league: str
    time: str
    probabilities: ProbabilitiesResponse
    predicted_outcome: str
    confidence: float
    odds: OddsResponse
    implied_probabilities: ImpliedProbsResponse
    expected_goals: ExpectedGoalsResponse | None = None
    over_under: OverUnderResponse | None = None
    btts: BttsResponse | None = None
    top_scorelines: list[ScorelineResponse] | None = None
    form: FormResponse | None = None
    value_bets: list[ValueBetResponse] = []


class LeaguePredictionsResponse(BaseModel):
    league_code: str
    league_name: str
    matches: list[MatchPredictionResponse]


@router.get("/dates", response_model=list[str])
async def get_available_dates(
    request: Request,
    _: Annotated[str, Depends(get_approved_user)],
) -> list[str]:
    """Return sorted fixture dates (DD/MM/YYYY) for supported leagues."""
    service = request.app.state.prediction_service
    return await asyncio.to_thread(service.get_available_dates)


@router.get("/predictions", response_model=list[LeaguePredictionsResponse])
async def get_all_predictions(
    request: Request,
    _: Annotated[str, Depends(get_approved_user)],
    date: str | None = Query(None, description="Date in DD/MM/YYYY format"),
) -> list[LeaguePredictionsResponse]:
    """Get predictions for all leagues with fixtures today."""
    service = request.app.state.prediction_service
    all_results = await asyncio.to_thread(
        service.get_all_leagues_predictions, target_date=date
    )
    return [_build_league_response(r) for r in all_results]


@router.get(
    "/predictions/{league_code}",
    response_model=LeaguePredictionsResponse,
)
async def get_league_predictions(
    league_code: str,
    request: Request,
    _: Annotated[str, Depends(get_approved_user)],
    date: str | None = Query(None, description="Date in DD/MM/YYYY format"),
) -> LeaguePredictionsResponse:
    """Get predictions for a specific league."""
    service = request.app.state.prediction_service
    result = await asyncio.to_thread(
        service.get_league_predictions, league_code.upper(), date
    )
    return _build_league_response(result)


@router.post("/predictions/refresh")
async def refresh_predictions(
    request: Request,
    _: Annotated[str, Depends(get_approved_user)],
    league_code: str | None = Query(None),
) -> dict[str, str]:
    """Invalidate cache and recompute predictions."""
    service = request.app.state.prediction_service
    service.invalidate_cache(league_code)
    return {"status": "cache_cleared", "league": league_code or "all"}


def _build_league_response(result) -> LeaguePredictionsResponse:
    """Convert internal LeaguePredictions to API response."""
    matches = []
    for fixture, pred, stats in zip(
        result.fixtures, result.predictions, result.match_stats
    ):
        bk_probs = fixture.implied_probabilities()

        # Find value bets for this match
        match_vbs = [
            vb
            for vb in result.value_bets
            if vb.home_team == pred.home_team and vb.away_team == pred.away_team
        ]

        match_resp = MatchPredictionResponse(
            home_team=pred.home_team,
            away_team=pred.away_team,
            league=fixture.league,
            time=fixture.time or "",
            probabilities=ProbabilitiesResponse(
                home_win=round(pred.home_win_prob, 4),
                draw=round(pred.draw_prob, 4),
                away_win=round(pred.away_win_prob, 4),
            ),
            predicted_outcome=pred.predicted_outcome,
            confidence=round(pred.confidence, 4),
            odds=OddsResponse(
                home=fixture.b365_home,
                draw=fixture.b365_draw,
                away=fixture.b365_away,
            ),
            implied_probabilities=ImpliedProbsResponse(
                home=round(bk_probs.get("home", 0), 4),
                draw=round(bk_probs.get("draw", 0), 4),
                away=round(bk_probs.get("away", 0), 4),
            ),
            value_bets=[
                ValueBetResponse(
                    outcome=vb.outcome,
                    ml_probability=round(vb.ml_probability, 4),
                    bookmaker_implied=round(vb.bookmaker_probability, 4),
                    edge=round(vb.edge, 4),
                    edge_pct=f"{vb.edge * 100:.1f}%",
                    best_odds=round(vb.best_odds, 2),
                    kelly_fraction=round(vb.kelly_fraction, 4),
                    confidence=vb.confidence,
                )
                for vb in match_vbs
            ],
        )

        if stats:
            match_resp.expected_goals = ExpectedGoalsResponse(
                home=round(stats.home_xg, 2),
                away=round(stats.away_xg, 2),
                total=round(stats.total_xg, 2),
            )
            match_resp.over_under = OverUnderResponse(
                over_15=round(stats.over15_prob, 3),
                over_25=round(stats.over25_prob, 3),
                over_35=round(stats.over35_prob, 3),
                under_25=round(stats.under25_prob, 3),
            )
            match_resp.btts = BttsResponse(
                yes=round(stats.btts_yes_prob, 3),
                no=round(stats.btts_no_prob, 3),
            )
            match_resp.top_scorelines = [
                ScorelineResponse(
                    score=f"{h}-{a}",
                    prob=round(p, 3),
                )
                for h, a, p in (stats.top_scorelines or [])[:5]
            ]
            match_resp.form = FormResponse(
                home=round(stats.home_form, 2),
                away=round(stats.away_form, 2),
            )

        matches.append(match_resp)

    return LeaguePredictionsResponse(
        league_code=result.league_code,
        league_name=result.league_name,
        matches=matches,
    )
