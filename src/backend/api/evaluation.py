from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


class ScoreRequest(BaseModel):
    match_date: str
    actual_results: dict[str, str]


@router.get("/predictions")
def get_stored_predictions(match_date: str = Query(default=None)) -> dict[str, Any]:
    """Return stored pre-match predictions for a given date."""
    from src.evaluation.evaluation_service import EvaluationService

    target = match_date or date.today().strftime("%Y-%m-%d")
    service = EvaluationService(storage_dir="output/evaluation")
    try:
        entries = service.load_predictions(match_date=target)
        return {"date": target, "predictions": [e.to_dict() for e in entries]}
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"No predictions stored for {target}."
        )


@router.post("/score")
def score_predictions(body: ScoreRequest) -> dict[str, Any]:
    """Score stored predictions against provided actual results."""
    from src.evaluation.evaluation_service import EvaluationService

    service = EvaluationService(storage_dir="output/evaluation")
    try:
        metrics = service.score(
            match_date=body.match_date,
            actual_results=body.actual_results,
        )
        return {"date": body.match_date, "metrics": metrics}
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions stored for {body.match_date}.",
        )
