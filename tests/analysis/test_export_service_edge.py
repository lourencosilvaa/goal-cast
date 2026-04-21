import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.export_service import ExportService
from src.models.predictor import MatchPrediction
from src.scrapers.fixtures_fetcher import Fixture


def _make_fixture(league: str = "Premier League") -> Fixture:
    return Fixture(
        division="E0",
        league=league,
        date="19/04/2026",
        time="15:00",
        home_team="Arsenal",
        away_team="Chelsea",
        b365_home=2.10,
        b365_draw=3.40,
        b365_away=3.50,
    )


def _make_prediction() -> MatchPrediction:
    return MatchPrediction(
        home_team="Arsenal",
        away_team="Chelsea",
        home_win_prob=0.52,
        draw_prob=0.26,
        away_win_prob=0.22,
        predicted_outcome="Home Win",
        confidence=0.52,
    )


class TestExportServiceEdgeCases:

    def test_to_excel_with_many_predictions(self, tmp_path):
        service = ExportService()
        preds = [_make_prediction()] * 20
        fixtures = [_make_fixture()] * 20
        out = tmp_path / "big_export.xlsx"
        service.to_excel(predictions=preds, fixtures=fixtures, value_bets=[], output_path=out)
        df = pd.read_excel(out, sheet_name="Predictions")
        assert len(df) == 20

    def test_to_csv_single_prediction(self):
        service = ExportService()
        result = service.to_csv(
            predictions=[_make_prediction()],
            fixtures=[_make_fixture()],
            value_bets=[],
        )
        lines = result.strip().split("\n")
        assert len(lines) == 2  # header + 1 row

    def test_to_excel_parent_dir_created_automatically(self, tmp_path):
        service = ExportService()
        nested = tmp_path / "deep" / "nested" / "report.xlsx"
        service.to_excel(
            predictions=[_make_prediction()],
            fixtures=[_make_fixture()],
            value_bets=[],
            output_path=nested,
        )
        assert nested.exists()

    def test_predictions_probabilities_rounded_to_4_decimals(self):
        service = ExportService()
        result = service.to_csv(
            predictions=[_make_prediction()],
            fixtures=[_make_fixture()],
            value_bets=[],
        )
        assert "0.52" in result
