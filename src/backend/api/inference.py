"""On-demand inference endpoint: runs the HF model directly for a given date."""

from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from src.backend.core.auth import get_current_user
from src.backend.services.inference_service import InferenceService

router = APIRouter(prefix="/api/predictions", tags=["inference"])

_LEAGUE_CODES = ["E0", "SP1", "D1", "I1", "F1", "P1"]
_CACHE_DIR = Path("datasets/cache")


def _load_teams_from_cache() -> dict[str, list[str]]:
    """Load teams from local CSV cache for the latest season."""
    result: dict[str, list[str]] = {}
    for league in _LEAGUE_CODES:
        # Find latest season file for this league
        files = sorted(_CACHE_DIR.glob(f"*_{league}.csv"), reverse=True)
        if not files:
            continue
        try:
            df = pd.read_csv(files[0], encoding="utf-8", usecols=["HomeTeam", "AwayTeam"])
            teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
            result[league] = teams
        except Exception:
            continue
    return result


def get_inference_service(request: Request) -> InferenceService:
    config = request.app.state.config
    return InferenceService(config)


class InferenceResponse(BaseModel):
    predictions: list[dict[str, Any]]
    date: str
    league_code: str | None


class CustomPredictRequest(BaseModel):
    home_team: str
    away_team: str
    league_code: str


class CustomPredictResponse(BaseModel):
    home_team: str
    away_team: str
    predicted_outcome: str
    confidence: float
    probabilities: dict[str, float]
    league: str


@router.post("/infer", response_model=InferenceResponse)
async def run_inference(
    user_id: Annotated[str, Depends(get_current_user)],
    inference_svc: Annotated[InferenceService, Depends(get_inference_service)],
    date: str | None = None,
    league_code: str | None = None,
) -> InferenceResponse:
    """Run on-demand ML predictions against the HuggingFace-hosted model."""
    league_codes = [league_code] if league_code else None
    try:
        predictions = await inference_svc.run(target_date=date, league_codes=league_codes)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    return InferenceResponse(
        predictions=predictions,
        date=date or "",
        league_code=league_code,
    )


@router.post("/custom", response_model=CustomPredictResponse)
async def predict_custom(
    body: CustomPredictRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    inference_svc: Annotated[InferenceService, Depends(get_inference_service)],
) -> CustomPredictResponse:
    """Run a one-off ML prediction for any chosen home/away team pair."""
    try:
        result = await inference_svc.predict_custom(
            home_team=body.home_team,
            away_team=body.away_team,
            league_code=body.league_code,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    return CustomPredictResponse(**result)


@router.get("/teams")
async def get_teams(
    user_id: Annotated[str, Depends(get_current_user)],
    inference_svc: Annotated[InferenceService, Depends(get_inference_service)],
) -> dict[str, list[str]]:
    """Return available team names grouped by league code."""
    try:
        data = await inference_svc.get_teams()
        # Validate: if all leagues have the same teams, the data is bad
        team_lists = list(data.values())
        if len(team_lists) >= 2 and all(t == team_lists[0] for t in team_lists[1:]):
            return _load_teams_from_cache()
        return data
    except Exception:
        # Fallback: load teams from local CSV cache
        return _load_teams_from_cache()
