"""On-demand inference endpoint: runs the HF model directly for a given date."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from src.backend.core.auth import get_current_user
from src.backend.services.inference_service import InferenceService, PredictionRefused

router = APIRouter(prefix="/api/predictions", tags=["inference"])


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
    #: The away side's league, when the two differ. Absent means the same one.
    away_league_code: str | None = None


class CustomPredictResponse(BaseModel):
    home_team: str
    away_team: str
    predicted_outcome: str
    confidence: float
    probabilities: dict[str, float]
    league: str
    #: Both default to ``None`` so an older Space — it deploys separately —
    #: keeps answering domestic requests while it catches up.
    away_league: str | None = None
    #: What produced the numbers. ``None`` is the domestic ensemble.
    model: str | None = None


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
    """Run a one-off ML prediction for any chosen home/away team pair.

    The pair may span two leagues, in which case the Space prices it with the
    cross-league model instead of the domestic ensemble and says so in
    ``model``.
    """
    try:
        result = await inference_svc.predict_custom(
            home_team=body.home_team,
            away_team=body.away_team,
            league_code=body.league_code,
            away_league_code=body.away_league_code,
        )
    # Before RuntimeError, which it subclasses: a stated refusal is an answer,
    # not an outage, and 503 would invite a retry that cannot ever succeed.
    except PredictionRefused as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    return CustomPredictResponse(**result)
