import csv
import io
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response

from src.backend.core.auth import get_approved_user

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("")
async def export_predictions(
    request: Request,
    _: Annotated[str, Depends(get_approved_user)],
    format: str = Query(default="csv", pattern="^(csv)$"),
    report_date: str | None = Query(default=None),
) -> Response:
    """
    Export predictions as CSV.

    Reads pre-computed predictions from Supabase (via PredictionService).
    Excel export has been removed to avoid heavy pandas/openpyxl dependency.
    """
    service = request.app.state.prediction_service

    # Use the requested date or today
    target_date = report_date or datetime.now().strftime("%d/%m/%Y")
    all_results = service.get_all_leagues_predictions(target_date=target_date)

    rows: list[dict[str, Any]] = []
    for league_pred in all_results:
        for m in league_pred.matches:
            probs = m.get("probabilities", {})
            odds = m.get("odds", {})
            vbs = m.get("value_bets", [])
            has_value = len(vbs) > 0
            rows.append(
                {
                    "league": m.get("league", league_pred.league_name),
                    "time": m.get("time", ""),
                    "home_team": m["home_team"],
                    "away_team": m["away_team"],
                    "predicted_outcome": m.get("predicted_outcome", ""),
                    "home_win_prob": probs.get("home_win", 0),
                    "draw_prob": probs.get("draw", 0),
                    "away_win_prob": probs.get("away_win", 0),
                    "confidence": m.get("confidence", 0),
                    "odds_home": odds.get("home", 0),
                    "odds_draw": odds.get("draw", 0),
                    "odds_away": odds.get("away", 0),
                    "value_bet": has_value,
                }
            )

    if not rows:
        raise HTTPException(
            status_code=404, detail="No predictions found for this date."
        )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

    timestamp = target_date.replace("/", "-")
    return PlainTextResponse(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=predictions_{timestamp}.csv"
        },
    )
