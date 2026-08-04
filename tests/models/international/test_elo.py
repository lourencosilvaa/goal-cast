import pandas as pd
import pytest

from config.config_loader import EloConfig
from src.models.international.elo import InternationalELO


@pytest.fixture
def elo_config() -> EloConfig:
    return EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0)


def _match(home, away, hg, ag, neutral, date) -> dict:
    return {
        "Date": pd.Timestamp(date),
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": hg,
        "FTAG": ag,
        "Neutral": neutral,
    }


class TestInternationalELO:
    def test_neutral_factor_zero_removes_home_advantage(self, elo_config):
        df = pd.DataFrame(
            [_match("A", "B", 1, 0, True, "2020-01-01")]
        )
        elo = InternationalELO(elo_config, neutral_home_advantage_factor=0.0)
        featured = elo.compute_elo_features(df)
        # Both start equal; on a neutral pitch expected home prob must be 0.5
        assert featured["elo_expected_home"].iloc[0] == pytest.approx(0.5, abs=1e-9)

    def test_non_neutral_keeps_home_advantage(self, elo_config):
        df = pd.DataFrame(
            [_match("A", "B", 1, 0, False, "2020-01-01")]
        )
        elo = InternationalELO(elo_config, neutral_home_advantage_factor=0.0)
        featured = elo.compute_elo_features(df)
        assert featured["elo_expected_home"].iloc[0] > 0.5

    def test_partial_neutral_factor(self, elo_config):
        df = pd.DataFrame(
            [_match("A", "B", 1, 0, True, "2020-01-01")]
        )
        elo = InternationalELO(elo_config, neutral_home_advantage_factor=1.0)
        featured = elo.compute_elo_features(df)
        # factor 1.0 => full home advantage even on neutral ground
        assert featured["elo_expected_home"].iloc[0] > 0.5

    def test_elo_feature_columns_present(self, elo_config):
        df = pd.DataFrame(
            [_match("A", "B", 2, 1, False, "2020-01-01")]
        )
        elo = InternationalELO(elo_config, neutral_home_advantage_factor=0.0)
        featured = elo.compute_elo_features(df)
        for col in (
            "elo_home",
            "elo_away",
            "elo_diff",
            "elo_expected_home",
            "elo_expected_away",
        ):
            assert col in featured.columns

    def test_away_win_updates_ratings(self, elo_config):
        df = pd.DataFrame([_match("A", "B", 0, 2, False, "2020-01-01")])
        elo = InternationalELO(elo_config, neutral_home_advantage_factor=0.0)
        elo.compute_elo_features(df)
        assert elo.get_rating("B") > elo.get_rating("A")

    def test_draw_updates_ratings_symmetrically(self, elo_config):
        df = pd.DataFrame([_match("A", "B", 1, 1, True, "2020-01-01")])
        elo = InternationalELO(elo_config, neutral_home_advantage_factor=0.0)
        elo.compute_elo_features(df)
        assert elo.get_rating("A") == pytest.approx(elo.get_rating("B"))
