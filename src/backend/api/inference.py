"""On-demand inference endpoint: runs the HF model directly for a given date."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from src.backend.core.auth import get_current_user
from src.backend.services.inference_service import InferenceService

router = APIRouter(prefix="/api/predictions", tags=["inference"])


def get_inference_service(request: Request) -> InferenceService:
    config = request.app.state.config
    return InferenceService(config)


class InferenceResponse(BaseModel):
    predictions: list[dict[str, Any]]
    date: str
    league_code: str | None


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
        predictions = inference_svc.run(target_date=date, league_codes=league_codes)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    return InferenceResponse(
        predictions=predictions,
        date=date or "",
        league_code=league_code,
    )
