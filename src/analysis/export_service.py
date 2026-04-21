from enum import Enum
from pathlib import Path

import pandas as pd

from src.analysis.value_detector import ValueBet
from src.models.predictor import MatchPrediction
from src.scrapers.fixtures_fetcher import Fixture


class ExportFormat(Enum):
    CSV = "csv"
    EXCEL = "excel"


class ExportService:
    """
    Exports predictions, value bets, and odds to CSV or Excel files.
    Excel output uses three sheets: Predictions, Value Bets, Odds.
    """

    def _build_predictions_df(
        self,
        predictions: list[MatchPrediction],
        fixtures: list[Fixture],
        value_bets: list[ValueBet],
    ) -> pd.DataFrame:
        value_set = {(vb.home_team, vb.away_team, vb.outcome) for vb in value_bets}
        rows = []
        for pred, fix in zip(predictions, fixtures):
            has_value = any(
                (pred.home_team, pred.away_team, o) in value_set
                for o in ["Home Win", "Draw", "Away Win"]
            )
            rows.append(
                {
                    "date": fix.date,
                    "time": fix.time,
                    "league": fix.league,
                    "home_team": pred.home_team,
                    "away_team": pred.away_team,
                    "predicted_outcome": pred.predicted_outcome,
                    "home_win_prob": round(pred.home_win_prob, 4),
                    "draw_prob": round(pred.draw_prob, 4),
                    "away_win_prob": round(pred.away_win_prob, 4),
                    "confidence": round(pred.confidence, 4),
                    "value_bet": has_value,
                }
            )
        return pd.DataFrame(rows)

    def _build_value_bets_df(self, value_bets: list[ValueBet]) -> pd.DataFrame:
        if not value_bets:
            return pd.DataFrame(
                columns=[
                    "home_team",
                    "away_team",
                    "outcome",
                    "ml_probability",
                    "bookmaker_probability",
                    "edge",
                    "best_odds",
                    "kelly_fraction",
                    "confidence",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "home_team": vb.home_team,
                    "away_team": vb.away_team,
                    "outcome": vb.outcome,
                    "ml_probability": round(vb.ml_probability, 4),
                    "bookmaker_probability": round(vb.bookmaker_probability, 4),
                    "edge": round(vb.edge, 4),
                    "best_odds": round(vb.best_odds, 2),
                    "kelly_fraction": round(vb.kelly_fraction, 4),
                    "confidence": vb.confidence,
                }
                for vb in value_bets
            ]
        )

    def _build_odds_df(self, fixtures: list[Fixture]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": fix.date,
                    "league": fix.league,
                    "home_team": fix.home_team,
                    "away_team": fix.away_team,
                    "b365_home": fix.b365_home,
                    "b365_draw": fix.b365_draw,
                    "b365_away": fix.b365_away,
                }
                for fix in fixtures
            ]
        )

    def to_csv(
        self,
        predictions: list[MatchPrediction],
        fixtures: list[Fixture],
        value_bets: list[ValueBet],
        output_path: Path | None = None,
    ) -> str:
        if not predictions:
            raise ValueError("Cannot export: predictions list is empty")

        df = self._build_predictions_df(predictions, fixtures, value_bets)
        csv_str = df.to_csv(index=False)

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(csv_str, encoding="utf-8")

        return csv_str

    def to_excel(
        self,
        predictions: list[MatchPrediction],
        fixtures: list[Fixture],
        value_bets: list[ValueBet],
        output_path: Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        predictions_df = self._build_predictions_df(predictions, fixtures, value_bets)
        value_bets_df = self._build_value_bets_df(value_bets)
        odds_df = self._build_odds_df(fixtures)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            predictions_df.to_excel(writer, sheet_name="Predictions", index=False)
            value_bets_df.to_excel(writer, sheet_name="Value Bets", index=False)
            odds_df.to_excel(writer, sheet_name="Odds", index=False)
