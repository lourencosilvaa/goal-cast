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
    """Invalidate in-memory cache (predictions are reloaded from Supabase)."""
    service = request.app.state.prediction_service
    service.invalidate_cache(league_code)
    return {"status": "cache_cleared", "league": league_code or "all"}


def _build_league_response(result) -> LeaguePredictionsResponse:
    """Convert LeaguePredictions (pre-built match dicts from Supabase) to API response."""
    matches = []
    for m in result.matches:
        probs = m.get("probabilities", {})
        odds = m.get("odds", {})
        imp = m.get("implied_probabilities", {})
        xg = m.get("expected_goals")
        ou = m.get("over_under")
        btts = m.get("btts")
        scorelines = m.get("top_scorelines")
        form = m.get("form")
        vbs = m.get("value_bets", [])

        match_resp = MatchPredictionResponse(
            home_team=m["home_team"],
            away_team=m["away_team"],
            league=m.get("league", ""),
            time=m.get("time", ""),
            probabilities=ProbabilitiesResponse(
                home_win=probs.get("home_win", 0),
                draw=probs.get("draw", 0),
                away_win=probs.get("away_win", 0),
            ),
            predicted_outcome=m.get("predicted_outcome", ""),
            confidence=m.get("confidence", 0),
            odds=OddsResponse(
                home=odds.get("home", 0),
                draw=odds.get("draw", 0),
                away=odds.get("away", 0),
            ),
            implied_probabilities=ImpliedProbsResponse(
                home=imp.get("home", 0),
                draw=imp.get("draw", 0),
                away=imp.get("away", 0),
            ),
            expected_goals=ExpectedGoalsResponse(**xg) if xg else None,
            over_under=OverUnderResponse(**ou) if ou else None,
            btts=BttsResponse(**btts) if btts else None,
            top_scorelines=(
                [ScorelineResponse(**s) for s in scorelines] if scorelines else None
            ),
            form=FormResponse(**form) if form else None,
            value_bets=[ValueBetResponse(**vb) for vb in vbs],
        )
        matches.append(match_resp)

    return LeaguePredictionsResponse(
        league_code=result.league_code,
        league_name=result.league_name,
        matches=matches,
    )
