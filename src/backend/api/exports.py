import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response

from src.backend.core.auth import get_approved_user

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("")
def export_predictions(
    _: Annotated[str, Depends(get_approved_user)],
    format: str = Query(default="csv", pattern="^(csv|excel)$"),
    report_date: str = Query(default=None),
) -> Response:
    """
    Export latest predictions as CSV or Excel.
    Reads from the most recent JSON report in output/reports/.
    """
    from src.analysis.export_service import ExportService
    from src.models.predictor import MatchPrediction
    from src.scrapers.fixtures_fetcher import Fixture

    report_path = Path("output/reports/value_report_latest.json")
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No prediction report found. Run find_value_bets.py first.",
        )

    try:
        data = json.loads(report_path.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read report: {e}")

    def _to_match_prediction(p: dict[str, Any]) -> MatchPrediction:
        probs = p.get("probabilities", {})
        return MatchPrediction(
            home_team=p["home_team"],
            away_team=p["away_team"],
            home_win_prob=probs.get("home_win", p.get("home_win_prob", 0.0)),
            draw_prob=probs.get("draw", p.get("draw_prob", 0.0)),
            away_win_prob=probs.get("away_win", p.get("away_win_prob", 0.0)),
            predicted_outcome=p["predicted_outcome"],
            confidence=p["confidence"],
        )

    predictions = [_to_match_prediction(p) for p in data.get("predictions", [])]

    def _to_fixture(f: dict[str, Any]) -> Fixture:
        odds = f.get("b365_odds", {})
        return Fixture(
            division=f.get("division", ""),
            league=f.get("league", ""),
            date=f.get("date", ""),
            time=f.get("time", ""),
            home_team=f["home_team"],
            away_team=f["away_team"],
            b365_home=odds.get("home", f.get("b365_home", 0.0)),
            b365_draw=odds.get("draw", f.get("b365_draw", 0.0)),
            b365_away=odds.get("away", f.get("b365_away", 0.0)),
        )

    fixtures_data = data.get("fixtures", [])
    fixtures = (
        [_to_fixture(f) for f in fixtures_data]
        if fixtures_data
        else [
            Fixture(
                division="",
                league="",
                date="",
                time="",
                home_team=p.home_team,
                away_team=p.away_team,
                b365_home=0.0,
                b365_draw=0.0,
                b365_away=0.0,
            )
            for p in predictions
        ]
    )
    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions in report.")

    service = ExportService()
    timestamp = report_date or date.today().strftime("%Y-%m-%d")

    if format == "csv":
        csv_str = service.to_csv(predictions, fixtures, [])
        return PlainTextResponse(
            content=csv_str,
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=predictions_{timestamp}.csv"
                )
            },
        )
    else:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        service.to_excel(predictions, fixtures, [], output_path=tmp_path)
        return FileResponse(
            path=tmp_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"predictions_{timestamp}.xlsx",
        )
