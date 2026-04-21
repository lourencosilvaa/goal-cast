from dataclasses import dataclass

from config.config_loader import AnalysisConfig
from src.models.predictor import MatchPrediction
from src.scrapers.odds_aggregator import AggregatedOdds


@dataclass
class ValueBet:
    """A detected value bet opportunity."""

    home_team: str
    away_team: str
    outcome: str
    ml_probability: float
    bookmaker_probability: float
    edge: float
    best_odds: float
    best_source: str
    kelly_fraction: float
    confidence: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        return {
            "match": f"{self.home_team} vs {self.away_team}",
            "recommended_bet": self.outcome,
            "ml_probability": round(self.ml_probability, 4),
            "bookmaker_implied": round(self.bookmaker_probability, 4),
            "edge": round(self.edge, 4),
            "edge_pct": f"{self.edge * 100:.1f}%",
            "best_odds": round(self.best_odds, 2),
            "best_source": self.best_source,
            "kelly_fraction": round(self.kelly_fraction, 4),
            "confidence": self.confidence,
        }


class ValueDetector:
    """
    Compares ML model probabilities with bookmaker odds
    to identify value betting opportunities.

    A value bet exists when: ML probability > bookmaker implied probability
    by a margin exceeding the configured threshold.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config

    def find_value_bets(
        self,
        prediction: MatchPrediction,
        aggregated_odds: AggregatedOdds,
    ) -> list[ValueBet]:
        """Find value bets for a single match."""
        value_bets: list[ValueBet] = []
        bk_probs = aggregated_odds.avg_implied_probabilities()

        outcomes = [
            (
                "Home Win",
                prediction.home_win_prob,
                bk_probs.get("home", 0),
                aggregated_odds.best_home_win,
                "home_win",
            ),
            (
                "Draw",
                prediction.draw_prob,
                bk_probs.get("draw", 0),
                aggregated_odds.best_draw,
                "draw",
            ),
            (
                "Away Win",
                prediction.away_win_prob,
                bk_probs.get("away", 0),
                aggregated_odds.best_away_win,
                "away_win",
            ),
        ]

        for outcome_name, ml_prob, bk_prob, best_odds, _ in outcomes:
            if bk_prob <= 0 or best_odds <= 1.0:
                continue

            edge = ml_prob - bk_prob

            if edge >= self.config.min_edge:
                kelly = self._kelly_criterion(ml_prob, best_odds)
                confidence = self._assess_confidence(edge, ml_prob)

                best_source = self._find_best_source(aggregated_odds, outcome_name)

                value_bets.append(
                    ValueBet(
                        home_team=prediction.home_team,
                        away_team=prediction.away_team,
                        outcome=outcome_name,
                        ml_probability=ml_prob,
                        bookmaker_probability=bk_prob,
                        edge=edge,
                        best_odds=best_odds,
                        best_source=best_source,
                        kelly_fraction=kelly,
                        confidence=confidence,
                    )
                )

        return sorted(value_bets, key=lambda v: -v.edge)

    def find_value_bets_batch(
        self,
        predictions: list[MatchPrediction],
        odds_list: list[AggregatedOdds],
    ) -> list[ValueBet]:
        """Find value bets across multiple matches."""
        all_value_bets: list[ValueBet] = []

        for prediction in predictions:
            matching_odds = self._find_matching_odds(prediction, odds_list)
            if matching_odds:
                bets = self.find_value_bets(prediction, matching_odds)
                all_value_bets.extend(bets)

        return sorted(all_value_bets, key=lambda v: -v.edge)

    @staticmethod
    def _kelly_criterion(probability: float, odds: float) -> float:
        """
        Kelly Criterion for optimal bet sizing.
        f = (bp - q) / b
        where b = odds - 1, p = probability, q = 1 - p
        """
        b = odds - 1
        q = 1 - probability

        if b <= 0:
            return 0.0

        kelly = (b * probability - q) / b
        return max(0.0, min(kelly, 0.25))  # Cap at 25% of bankroll

    @staticmethod
    def _assess_confidence(edge: float, ml_prob: float) -> str:
        """Assess confidence level of the value bet."""
        if edge >= 0.10 and ml_prob >= 0.50:
            return "HIGH"
        elif edge >= 0.05 and ml_prob >= 0.35:
            return "MEDIUM"
        else:
            return "LOW"

    @staticmethod
    def _find_best_source(aggregated: AggregatedOdds, outcome: str) -> str:
        """Find which source offers the best odds for an outcome."""
        best_odds = 0.0
        best_src = ""

        for source in aggregated.sources:
            if outcome == "Home Win" and source.home_win > best_odds:
                best_odds = source.home_win
                best_src = source.source
            elif outcome == "Draw" and source.draw > best_odds:
                best_odds = source.draw
                best_src = source.source
            elif outcome == "Away Win" and source.away_win > best_odds:
                best_odds = source.away_win
                best_src = source.source

        return best_src

    @staticmethod
    def _find_matching_odds(
        prediction: MatchPrediction,
        odds_list: list[AggregatedOdds],
    ) -> AggregatedOdds | None:
        """Find matching aggregated odds for a prediction."""
        home_lower = prediction.home_team.lower().strip()
        away_lower = prediction.away_team.lower().strip()

        for odds in odds_list:
            if (
                home_lower in odds.home_team.lower()
                or odds.home_team.lower() in home_lower
            ) and (
                away_lower in odds.away_team.lower()
                or odds.away_team.lower() in away_lower
            ):
                return odds

        return None
