import json
from datetime import datetime, timezone
from pathlib import Path

from src.analysis.value_detector import ValueBet
from src.models.predictor import MatchPrediction
from src.scrapers.odds_aggregator import AggregatedOdds


class ReportGenerator:
    """Generates structured JSON and text reports for the Copilot agent."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_prediction_report(
        self,
        predictions: list[MatchPrediction],
        odds: list[AggregatedOdds],
        value_bets: list[ValueBet],
    ) -> dict[str, object]:
        """Generate a full prediction report as JSON."""
        report: dict[str, object] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_matches": len(predictions),
                "total_value_bets": len(value_bets),
                "high_confidence": sum(1 for v in value_bets if v.confidence == "HIGH"),
                "medium_confidence": sum(
                    1 for v in value_bets if v.confidence == "MEDIUM"
                ),
            },
            "predictions": [p.to_dict() for p in predictions],
            "odds": [o.to_dict() for o in odds],
            "value_bets": [v.to_dict() for v in value_bets],
        }

        return report

    def save_report(
        self,
        report: dict[str, object],
        filename: str | None = None,
    ) -> Path:
        """Save report to JSON file."""
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.json"

        filepath = self.output_dir / filename
        with filepath.open("w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"Report saved to {filepath}")
        return filepath

    def format_text_report(
        self,
        predictions: list[MatchPrediction],
        value_bets: list[ValueBet],
    ) -> str:
        """Format a human-readable text report."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  FOOTBALL PREDICTION REPORT")
        lines.append(
            f"  Generated: "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        lines.append("=" * 60)

        lines.append(f"\n  Total Matches Analyzed: {len(predictions)}")
        lines.append(f"  Value Bets Found: {len(value_bets)}")

        if predictions:
            lines.append("\n" + "-" * 60)
            lines.append("  MATCH PREDICTIONS")
            lines.append("-" * 60)

            for pred in predictions:
                lines.append(f"\n  {pred.home_team} vs {pred.away_team}")
                lines.append(
                    f"    H: {pred.home_win_prob:.1%} | "
                    f"D: {pred.draw_prob:.1%} | "
                    f"A: {pred.away_win_prob:.1%}"
                )
                lines.append(
                    f"    Prediction: {pred.predicted_outcome} "
                    f"({pred.confidence:.1%} confidence)"
                )

        if value_bets:
            lines.append("\n" + "-" * 60)
            lines.append("  VALUE BETS")
            lines.append("-" * 60)

            for vb in value_bets:
                lines.append(f"\n  {vb.home_team} vs {vb.away_team} " f"- {vb.outcome}")
                lines.append(
                    f"    ML Prob: {vb.ml_probability:.1%} vs "
                    f"Bookmaker: {vb.bookmaker_probability:.1%}"
                )
                lines.append(
                    f"    Edge: {vb.edge:.1%} | "
                    f"Best Odds: {vb.best_odds:.2f} ({vb.best_source})"
                )
                lines.append(
                    f"    Kelly: {vb.kelly_fraction:.1%} | "
                    f"Confidence: {vb.confidence}"
                )

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
