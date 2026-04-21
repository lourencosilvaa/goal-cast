import pandas as pd

from config.config_loader import EloConfig
from src.models.elo import FootballELO


class TestFootballELO:

    def _make_config(self) -> EloConfig:
        return EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0)

    def test_initial_rating(self):
        elo = FootballELO(self._make_config())
        assert elo.get_rating("Arsenal") == 1500.0

    def test_expected_score_equal_ratings(self):
        elo = FootballELO(self._make_config())
        score = elo.expected_score(1500.0, 1500.0)
        assert abs(score - 0.5) < 0.001

    def test_expected_score_higher_rated_wins_more(self):
        elo = FootballELO(self._make_config())
        score = elo.expected_score(1600.0, 1400.0)
        assert score > 0.5

    def test_update_winner_gains_rating(self):
        elo = FootballELO(self._make_config())
        initial = elo.get_rating("Arsenal")
        elo.update("Arsenal", "Chelsea", 3, 0)
        assert elo.get_rating("Arsenal") > initial

    def test_update_loser_loses_rating(self):
        elo = FootballELO(self._make_config())
        initial = elo.get_rating("Chelsea")
        elo.update("Arsenal", "Chelsea", 3, 0)
        assert elo.get_rating("Chelsea") < initial

    def test_update_draw_balanced(self):
        elo = FootballELO(self._make_config())
        elo.update("Arsenal", "Chelsea", 1, 1)
        # Home team gets slight disadvantage because of home_advantage in calculation
        # but both should be close to initial
        assert abs(elo.get_rating("Arsenal") - 1500.0) < 30
        assert abs(elo.get_rating("Chelsea") - 1500.0) < 30

    def test_margin_multiplier_larger_diff_more_impact(self):
        elo = FootballELO(self._make_config())
        small = elo.margin_multiplier(1)
        large = elo.margin_multiplier(5)
        assert large > small

    def test_compute_elo_features_adds_columns(self, sample_match_data):
        from src.models.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        clean = cleaner.clean(sample_match_data)

        elo = FootballELO(self._make_config())
        result = elo.compute_elo_features(clean)

        assert "elo_home" in result.columns
        assert "elo_away" in result.columns
        assert "elo_diff" in result.columns
        assert "elo_expected_home" in result.columns

    def test_compute_elo_features_preserves_row_count(self, sample_match_data):
        from src.models.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        clean = cleaner.clean(sample_match_data)

        elo = FootballELO(self._make_config())
        result = elo.compute_elo_features(clean)

        assert len(result) == len(clean)

    def test_get_top_teams(self):
        elo = FootballELO(self._make_config())
        elo.update("Arsenal", "Chelsea", 3, 0)
        elo.update("Arsenal", "Liverpool", 2, 1)
        elo.update("Man City", "Chelsea", 4, 0)

        top = elo.get_top_teams(2)
        assert len(top) == 2
        assert top[0][1] >= top[1][1]
