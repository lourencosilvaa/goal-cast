from config.config_loader import AnalysisConfig
from src.analysis.value_detector import ValueDetector
from src.models.predictor import MatchPrediction
from src.scrapers.base_scraper import ScrapedOdds
from src.scrapers.odds_aggregator import AggregatedOdds


class TestValueDetector:

    def _make_config(self) -> AnalysisConfig:
        return AnalysisConfig(
            value_threshold=0.05,
            min_edge=0.03,
            blend_weights={"ml_model": 0.50, "bookmaker_avg": 0.30, "best_odds": 0.20},
        )

    def _make_prediction(
        self, home_prob: float = 0.55, draw_prob: float = 0.25, away_prob: float = 0.20
    ) -> MatchPrediction:
        return MatchPrediction(
            home_team="Arsenal",
            away_team="Chelsea",
            home_win_prob=home_prob,
            draw_prob=draw_prob,
            away_win_prob=away_prob,
            predicted_outcome="Home Win",
            confidence=home_prob,
        )

    def _make_odds(
        self, home: float = 2.00, draw: float = 3.50, away: float = 4.00
    ) -> AggregatedOdds:
        source = ScrapedOdds(
            source="Betclic", home_team="Arsenal", away_team="Chelsea",
            home_win=home, draw=draw, away_win=away,
            league="Premier League", match_date="2024-01-01", url="",
        )
        return AggregatedOdds(
            home_team="Arsenal", away_team="Chelsea",
            league="Premier League", match_date="2024-01-01",
            sources=[source],
        )

    def test_find_value_bet_with_edge(self):
        detector = ValueDetector(self._make_config())
        # ML says 55% home, bookmaker implies ~50% (odds 2.00)
        pred = self._make_prediction(home_prob=0.55)
        odds = self._make_odds(home=2.00)
        value_bets = detector.find_value_bets(pred, odds)

        assert len(value_bets) >= 1
        home_bet = [v for v in value_bets if v.outcome == "Home Win"]
        assert len(home_bet) == 1
        assert home_bet[0].edge > 0

    def test_no_value_bet_when_no_edge(self):
        detector = ValueDetector(self._make_config())
        # ML says 40% home, bookmaker implies ~50% (odds 2.00) → no value
        pred = self._make_prediction(home_prob=0.40, draw_prob=0.30, away_prob=0.30)
        odds = self._make_odds(home=2.00)
        value_bets = detector.find_value_bets(pred, odds)

        home_bets = [v for v in value_bets if v.outcome == "Home Win"]
        assert len(home_bets) == 0

    def test_kelly_criterion_positive_edge(self):
        kelly = ValueDetector._kelly_criterion(0.60, 2.50)
        assert kelly > 0
        assert kelly <= 0.25  # Capped

    def test_kelly_criterion_no_edge(self):
        kelly = ValueDetector._kelly_criterion(0.30, 2.00)
        assert kelly == 0.0

    def test_confidence_high(self):
        confidence = ValueDetector._assess_confidence(0.12, 0.55)
        assert confidence == "HIGH"

    def test_confidence_medium(self):
        confidence = ValueDetector._assess_confidence(0.06, 0.40)
        assert confidence == "MEDIUM"

    def test_confidence_low(self):
        confidence = ValueDetector._assess_confidence(0.03, 0.30)
        assert confidence == "LOW"

    def test_value_bet_to_dict(self):
        detector = ValueDetector(self._make_config())
        pred = self._make_prediction(home_prob=0.60)
        odds = self._make_odds(home=2.00)
        value_bets = detector.find_value_bets(pred, odds)

        assert len(value_bets) > 0
        d = value_bets[0].to_dict()
        assert "match" in d
        assert "edge" in d
        assert "kelly_fraction" in d
        assert "confidence" in d

    def test_value_bets_sorted_by_edge(self):
        detector = ValueDetector(self._make_config())
        pred = self._make_prediction(home_prob=0.55, draw_prob=0.30, away_prob=0.35)
        odds = self._make_odds(home=2.00, draw=3.50, away=4.00)
        value_bets = detector.find_value_bets(pred, odds)

        if len(value_bets) >= 2:
            for i in range(len(value_bets) - 1):
                assert value_bets[i].edge >= value_bets[i + 1].edge
