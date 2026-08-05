import numpy as np
import pandas as pd
import pytest

from config.config_loader import (
    EloConfig,
    FatigueConfig,
    FeaturesConfig,
    XgProxyConfig,
)
from src.models.international.base import AbstractFeatureEngineer
from src.models.international.feature_engineer import InternationalFeatureEngineer


@pytest.fixture
def features_config() -> FeaturesConfig:
    return FeaturesConfig(
        rolling_window=5,
        elo=EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0),
        xg_proxy=XgProxyConfig(sot_conversion=0.3, shot_conversion=0.03),
        fatigue=FatigueConfig(
            max_rest_days=30, default_rest_days=14, fatigue_threshold=3
        ),
    )


@pytest.fixture
def international_matches() -> pd.DataFrame:
    """Goals-only international matches (no shots/cards/odds columns)."""
    teams = ["Portugal", "Spain", "France", "Brazil"]
    rows: list[dict] = []
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2015-01-01")
    for i in range(60):
        h, a = rng.choice(teams, size=2, replace=False)
        hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        ftr = "H" if hg > ag else ("A" if ag > hg else "D")
        rows.append(
            {
                "Date": base + pd.Timedelta(days=10 * i),
                "HomeTeam": h,
                "AwayTeam": a,
                "FTHG": hg,
                "FTAG": ag,
                "FTR": ftr,
                "Neutral": bool(i % 3 == 0),
                "Tournament": "Friendly",
            }
        )
    return pd.DataFrame(rows)


class TestInternationalFeatureEngineer:
    def test_implements_abstract_engineer(self, features_config):
        eng = InternationalFeatureEngineer(features_config)
        assert isinstance(eng, AbstractFeatureEngineer)

    def test_produces_goal_based_features(
        self, features_config, international_matches
    ):
        eng = InternationalFeatureEngineer(features_config)
        out = eng.build_all_features(international_matches)
        for col in (
            "home_avg_GF",
            "away_avg_GF",
            "diff_avg_GF",
            "home_avg_GA",
            "away_avg_GA",
            "home_Form",
            "away_Form",
        ):
            assert col in out.columns

    def test_adds_neutral_flag_feature(self, features_config, international_matches):
        eng = InternationalFeatureEngineer(features_config)
        out = eng.build_all_features(international_matches)
        assert "is_neutral" in out.columns
        assert set(out["is_neutral"].unique()) <= {0, 1}

    def test_does_not_require_shot_columns(
        self, features_config, international_matches
    ):
        # No HS/AS/HST/... columns exist; must not raise
        eng = InternationalFeatureEngineer(features_config)
        out = eng.build_all_features(international_matches)
        assert not out.empty

    def test_no_odds_or_shot_features_created(
        self, features_config, international_matches
    ):
        eng = InternationalFeatureEngineer(features_config)
        out = eng.build_all_features(international_matches)
        assert "norm_prob_H" not in out.columns
        assert "home_xG_rolling" not in out.columns

    def test_h2h_and_draw_features_present(
        self, features_config, international_matches
    ):
        eng = InternationalFeatureEngineer(features_config)
        out = eng.build_all_features(international_matches)
        assert "h2h_home_wins" in out.columns
        assert "avg_draw_pct" in out.columns
