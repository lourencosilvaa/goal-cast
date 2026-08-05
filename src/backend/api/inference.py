"""On-demand inference endpoint: runs the HF model directly for a given date."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from src.backend.core.auth import get_current_user
from src.backend.repositories.team_repository import (
    FallbackTeamRepository,
    RemoteTeamRepository,
    StaticTeamRepository,
    TeamRepository,
)
from src.backend.services.inference_service import InferenceService

router = APIRouter(prefix="/api/predictions", tags=["inference"])


def get_inference_service(request: Request) -> InferenceService:
    config = request.app.state.config
    return InferenceService(config)


def get_team_repository(request: Request) -> TeamRepository:
    """Space first (freshest), shipped registry second (always available)."""
    config = request.app.state.config
    return FallbackTeamRepository(
        [
            RemoteTeamRepository(get_inference_service(request)),
            StaticTeamRepository(config.teams.registry_path),
        ]
    )


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
        predictions = await inference_svc.run(
            target_date=date, league_codes=league_codes
        )
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
    team_repo: Annotated[TeamRepository, Depends(get_team_repository)],
) -> dict[str, list[str]]:
    """Return available team names grouped by league code."""
    return await team_repo.get_teams()
