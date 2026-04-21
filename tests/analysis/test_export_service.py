import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.export_service import ExportFormat, ExportService
from src.analysis.value_detector import ValueBet
from src.models.predictor import MatchPrediction
from src.scrapers.fixtures_fetcher import Fixture


def _make_fixture() -> Fixture:
    return Fixture(
        division="E0",
        league="Premier League",
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


def _make_value_bet() -> ValueBet:
    return ValueBet(
        home_team="Arsenal",
        away_team="Chelsea",
        outcome="Home Win",
        ml_probability=0.52,
        bookmaker_probability=0.44,
        edge=0.08,
        best_odds=2.10,
        best_source="Bet365",
        kelly_fraction=0.08,
        confidence="HIGH",
    )


class TestExportService:

    def _make_service(self) -> ExportService:
        return ExportService()

    def test_to_csv_raises_on_empty_predictions(self):
        service = self._make_service()
        with pytest.raises(ValueError, match="empty"):
            service.to_csv(predictions=[], fixtures=[], value_bets=[])

    def test_to_csv_returns_string_with_expected_columns(self):
        service = self._make_service()
        result = service.to_csv(
            predictions=[_make_prediction()],
            fixtures=[_make_fixture()],
            value_bets=[],
        )
        assert isinstance(result, str)
        assert "Arsenal" in result
        assert "Chelsea" in result
        assert "home_win_prob" in result

    def test_to_csv_includes_value_bet_flag_when_present(self):
        service = self._make_service()
        result = service.to_csv(
            predictions=[_make_prediction()],
            fixtures=[_make_fixture()],
            value_bets=[_make_value_bet()],
        )
        assert "value_bet" in result

    def test_to_excel_creates_file_with_three_sheets(self, tmp_path):
        service = self._make_service()
        out_path = tmp_path / "export.xlsx"
        service.to_excel(
            predictions=[_make_prediction()],
            fixtures=[_make_fixture()],
            value_bets=[_make_value_bet()],
            output_path=out_path,
        )
        assert out_path.exists()
        wb = pd.ExcelFile(out_path)
        assert "Predictions" in wb.sheet_names
        assert "Value Bets" in wb.sheet_names
        assert "Odds" in wb.sheet_names

    def test_to_excel_value_bets_sheet_empty_when_no_bets(self, tmp_path):
        service = self._make_service()
        out_path = tmp_path / "export.xlsx"
        service.to_excel(
            predictions=[_make_prediction()],
            fixtures=[_make_fixture()],
            value_bets=[],
            output_path=out_path,
        )
        wb = pd.ExcelFile(out_path)
        df = pd.read_excel(out_path, sheet_name="Value Bets")
        assert df.empty

    def test_export_format_enum_values(self):
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.EXCEL.value == "excel"

    def test_to_csv_writes_to_file_when_path_provided(self, tmp_path):
        service = self._make_service()
        out_path = tmp_path / "report.csv"
        service.to_csv(
            predictions=[_make_prediction()],
            fixtures=[_make_fixture()],
            value_bets=[],
            output_path=out_path,
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
