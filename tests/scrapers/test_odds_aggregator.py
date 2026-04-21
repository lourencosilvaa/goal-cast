from src.scrapers.base_scraper import ScrapedOdds
from src.scrapers.odds_aggregator import AggregatedOdds


class TestAggregatedOdds:

    def _make_sources(self) -> list[ScrapedOdds]:
        return [
            ScrapedOdds(
                source="Betclic", home_team="Benfica", away_team="Porto",
                home_win=1.80, draw=3.50, away_win=4.20,
                league="Liga Portugal", match_date="2024-01-01", url="",
            ),
            ScrapedOdds(
                source="Betano", home_team="Benfica", away_team="Porto",
                home_win=1.85, draw=3.40, away_win=4.00,
                league="Liga Portugal", match_date="2024-01-01", url="",
            ),
            ScrapedOdds(
                source="Solverde", home_team="Benfica", away_team="Porto",
                home_win=1.75, draw=3.60, away_win=4.50,
                league="Liga Portugal", match_date="2024-01-01", url="",
            ),
        ]

    def test_avg_home_win(self):
        agg = AggregatedOdds(
            home_team="Benfica", away_team="Porto",
            league="Liga Portugal", match_date="2024-01-01",
            sources=self._make_sources(),
        )
        expected = (1.80 + 1.85 + 1.75) / 3
        assert abs(agg.avg_home_win - expected) < 0.01

    def test_best_home_win(self):
        agg = AggregatedOdds(
            home_team="Benfica", away_team="Porto",
            league="Liga Portugal", match_date="2024-01-01",
            sources=self._make_sources(),
        )
        assert agg.best_home_win == 1.85

    def test_best_away_win(self):
        agg = AggregatedOdds(
            home_team="Benfica", away_team="Porto",
            league="Liga Portugal", match_date="2024-01-01",
            sources=self._make_sources(),
        )
        assert agg.best_away_win == 4.50

    def test_avg_implied_probabilities_sum_to_one(self):
        agg = AggregatedOdds(
            home_team="Benfica", away_team="Porto",
            league="Liga Portugal", match_date="2024-01-01",
            sources=self._make_sources(),
        )
        probs = agg.avg_implied_probabilities()
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_to_dict_has_required_keys(self):
        agg = AggregatedOdds(
            home_team="Benfica", away_team="Porto",
            league="Liga Portugal", match_date="2024-01-01",
            sources=self._make_sources(),
        )
        d = agg.to_dict()
        assert "average_odds" in d
        assert "best_odds" in d
        assert "implied_probabilities" in d
        assert d["sources_count"] == 3

    def test_empty_sources(self):
        agg = AggregatedOdds(
            home_team="A", away_team="B",
            league="", match_date="",
            sources=[],
        )
        assert agg.avg_home_win == 0
        assert agg.best_home_win == 0
        probs = agg.avg_implied_probabilities()
        assert probs == {"home": 0, "draw": 0, "away": 0}
