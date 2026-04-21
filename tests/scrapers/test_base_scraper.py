from src.scrapers.base_scraper import ScrapedOdds


class TestScrapedOdds:

    def test_implied_probabilities_sum_to_one(self):
        odds = ScrapedOdds(
            source="Test",
            home_team="A",
            away_team="B",
            home_win=2.0,
            draw=3.5,
            away_win=4.0,
            league="Test League",
            match_date="2024-01-01",
            url="https://example.com",
        )
        probs = odds.implied_probabilities()
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.001

    def test_implied_probabilities_favorite_has_highest(self):
        odds = ScrapedOdds(
            source="Test",
            home_team="A",
            away_team="B",
            home_win=1.50,
            draw=4.00,
            away_win=6.00,
            league="Test League",
            match_date="2024-01-01",
            url="https://example.com",
        )
        probs = odds.implied_probabilities()
        assert probs["home"] > probs["draw"]
        assert probs["home"] > probs["away"]

    def test_to_dict_has_required_keys(self):
        odds = ScrapedOdds(
            source="Betclic",
            home_team="Benfica",
            away_team="Porto",
            home_win=1.80,
            draw=3.50,
            away_win=4.20,
            league="Liga Portugal",
            match_date="2024-01-01",
            url="https://example.com",
        )
        d = odds.to_dict()
        assert "source" in d
        assert "odds" in d
        assert "implied_probabilities" in d
        assert d["source"] == "Betclic"

    def test_implied_probabilities_zero_odds(self):
        odds = ScrapedOdds(
            source="Test",
            home_team="A",
            away_team="B",
            home_win=0,
            draw=0,
            away_win=0,
            league="Test",
            match_date="",
            url="",
        )
        probs = odds.implied_probabilities()
        assert probs == {"home": 0, "draw": 0, "away": 0}
