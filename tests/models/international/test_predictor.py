import pandas as pd
import pytest

from config.config_loader import EloConfig
from src.models.international.elo import InternationalELO
from src.models.international.predictor import (
    InternationalMatchPredictor,
    _models_available,
)
from src.models.predictor import MatchPrediction


class _FakePredictor:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.captured: pd.DataFrame | None = None

    def predict(self, df: pd.DataFrame) -> list[MatchPrediction]:
        self.captured = df
        row = df.iloc[0]
        return [
            MatchPrediction(
                home_team=str(row["HomeTeam"]),
                away_team=str(row["AwayTeam"]),
                home_win_prob=0.5,
                draw_prob=0.3,
                away_win_prob=0.2,
                predicted_outcome="Home Win",
                confidence=0.5,
            )
        ]


class _EmptyPredictor(_FakePredictor):
    def predict(self, df: pd.DataFrame) -> list[MatchPrediction]:
        self.captured = df
        return []


def _featured() -> pd.DataFrame:
    import numpy as np

    return pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-06-01"),
                "HomeTeam": "Portugal",
                "AwayTeam": "France",
                "home_avg_GF": 2.0,
                "away_avg_GF": 1.0,
                "home_avg_GA": 0.5,
                "away_avg_GA": 1.5,
                "home_Form": 2.5,
                "away_Form": 1.0,
                "home_draw_pct": 0.2,
                "away_draw_pct": 0.3,
                "h2h_home_wins": 0.6,
                "h2h_draws": 0.2,
                "nan_feat": np.nan,
                "Neutral": False,
            },
            {
                "Date": pd.Timestamp("2024-06-05"),
                "HomeTeam": "Spain",
                "AwayTeam": "Italy",
                "home_avg_GF": 1.8,
                "away_avg_GF": 1.2,
                "home_avg_GA": 0.7,
                "away_avg_GA": 1.1,
                "home_Form": 2.0,
                "away_Form": 1.5,
                "home_draw_pct": 0.25,
                "away_draw_pct": 0.2,
                "h2h_home_wins": 0.4,
                "h2h_draws": 0.3,
                "nan_feat": np.nan,
                "Neutral": False,
            },
        ]
    )


def _build_predictor(feature_names, neutral_factor: float = 0.0, featured=None):
    predictor = InternationalMatchPredictor.__new__(InternationalMatchPredictor)
    predictor._featured = _featured() if featured is None else featured
    elo = InternationalELO(
        EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0),
        neutral_home_advantage_factor=neutral_factor,
    )
    elo.ratings = {"Portugal": 1600.0, "Spain": 1550.0}
    predictor._elo = elo
    predictor._predictor = _FakePredictor(feature_names)
    return predictor


class TestInternationalMatchPredictor:
    def test_sets_is_neutral_flag(self):
        predictor = _build_predictor(["is_neutral", "home_avg_GF", "away_avg_GF"])
        predictor.predict("Portugal", "Spain", neutral=True)
        captured = predictor._predictor.captured
        assert int(captured["is_neutral"].iloc[0]) == 1

    def test_home_away_swap_reads_own_stats(self):
        # Spain played as HOME last, but is the AWAY team in this fixture,
        # so its own avg_GF must come from the home_avg_GF column (1.8).
        predictor = _build_predictor(["home_avg_GF", "away_avg_GF"])
        predictor.predict("Portugal", "Spain")
        captured = predictor._predictor.captured
        assert captured["home_avg_GF"].iloc[0] == pytest.approx(2.0)
        assert captured["away_avg_GF"].iloc[0] == pytest.approx(1.8)

    def test_elo_expectation_uses_neutral_flag(self):
        names = ["elo_expected_home", "elo_expected_away", "elo_diff"]
        neutral_pred = _build_predictor(names, neutral_factor=0.0)
        neutral_pred.predict("Portugal", "Spain", neutral=True)
        e_neutral = neutral_pred._predictor.captured["elo_expected_home"].iloc[0]

        home_pred = _build_predictor(names, neutral_factor=0.0)
        home_pred.predict("Portugal", "Spain", neutral=False)
        e_home = home_pred._predictor.captured["elo_expected_home"].iloc[0]

        assert e_home > e_neutral  # home advantage lifts expectation

    def test_diff_feature_computed(self):
        predictor = _build_predictor(["diff_avg_GF"])
        predictor.predict("Portugal", "Spain")
        captured = predictor._predictor.captured
        assert captured["diff_avg_GF"].iloc[0] == pytest.approx(2.0 - 1.8)

    def test_unknown_team_raises(self):
        predictor = _build_predictor(["home_avg_GF"])
        with pytest.raises(ValueError):
            predictor.predict("Portugal", "Atlantis")

    def test_models_available_false_when_missing(self, tmp_path):
        assert _models_available(tmp_path) is False

    def test_models_available_true_when_present(self, tmp_path):
        (tmp_path / "ensemble_model.joblib").write_text("x")
        assert _models_available(tmp_path) is True

    def test_all_derived_feature_branches(self):
        names = [
            "form_gap",
            "avg_draw_pct",
            "attack_similarity",
            "defense_similarity",
            "combined_defensive",
            "nan_feat",
            "unknown_feat",
        ]
        predictor = _build_predictor(names)
        predictor.predict("Portugal", "Spain")
        row = predictor._predictor.captured.iloc[0]
        # form_gap = |2.5 - 2.0| (Spain read from home_Form as it played home)
        assert row["form_gap"] == pytest.approx(0.5)
        assert row["avg_draw_pct"] == pytest.approx((0.2 + 0.25) / 2)
        assert row["attack_similarity"] == pytest.approx(1 / (1 + abs(2.0 - 1.8)))
        assert row["defense_similarity"] == pytest.approx(1 / (1 + abs(0.5 - 0.7)))
        assert row["combined_defensive"] == pytest.approx(
            1 / (1 + 0.5) + 1 / (1 + 0.7)
        )
        assert row["nan_feat"] == 0
        assert row["unknown_feat"] == 0

    def test_h2h_normal_branch_when_home(self):
        # Last Portugal-Spain meeting has Portugal at home -> direct read.
        featured = pd.DataFrame(
            [
                {
                    "Date": pd.Timestamp("2024-06-01"),
                    "HomeTeam": "Portugal",
                    "AwayTeam": "Spain",
                    "home_avg_GF": 2.0,
                    "away_avg_GF": 1.0,
                    "home_avg_GA": 0.5,
                    "away_avg_GA": 1.0,
                    "home_Form": 2.0,
                    "away_Form": 1.0,
                    "home_draw_pct": 0.2,
                    "away_draw_pct": 0.2,
                    "h2h_home_wins": 0.7,
                    "h2h_draws": 0.1,
                    "Neutral": False,
                }
            ]
        )
        predictor = _build_predictor(["h2h_home_wins"], featured=featured)
        predictor.predict("Portugal", "Spain")
        row = predictor._predictor.captured.iloc[0]
        assert row["h2h_home_wins"] == pytest.approx(0.7)

    def test_h2h_inverted_branch_when_away(self):
        # Last Portugal-Spain meeting has Spain at home; predicting Portugal
        # as home must invert the stored home-win rate.
        featured = pd.DataFrame(
            [
                {
                    "Date": pd.Timestamp("2024-06-01"),
                    "HomeTeam": "Spain",
                    "AwayTeam": "Portugal",
                    "home_avg_GF": 1.8,
                    "away_avg_GF": 1.2,
                    "home_avg_GA": 0.7,
                    "away_avg_GA": 1.1,
                    "home_Form": 2.0,
                    "away_Form": 1.0,
                    "home_draw_pct": 0.2,
                    "away_draw_pct": 0.2,
                    "h2h_home_wins": 0.6,
                    "h2h_draws": 0.1,
                    "Neutral": False,
                }
            ]
        )
        predictor = _build_predictor(["h2h_home_wins"], featured=featured)
        predictor.predict("Portugal", "Spain")
        row = predictor._predictor.captured.iloc[0]
        # inverted: max(0, 1 - 0.6 - 0.1)
        assert row["h2h_home_wins"] == pytest.approx(0.3)

    def test_h2h_zero_when_no_prior_meeting(self):
        predictor = _build_predictor(["h2h_home_wins"])  # Portugal never met Spain
        predictor.predict("Portugal", "Spain")
        assert predictor._predictor.captured["h2h_home_wins"].iloc[0] == 0

    def test_auto_prepare_when_predictor_missing(self):
        predictor = _build_predictor(["home_avg_GF"])
        fake = predictor._predictor
        predictor._predictor = None

        def _fake_prepare():
            predictor._predictor = fake
            return predictor

        predictor.prepare = _fake_prepare  # type: ignore[method-assign]
        result = predictor.predict("Portugal", "Spain")
        assert result.home_team == "Portugal"

    def test_empty_prediction_raises(self):
        predictor = _build_predictor(["home_avg_GF"])
        predictor._predictor = _EmptyPredictor(["home_avg_GF"])
        with pytest.raises(ValueError):
            predictor.predict("Portugal", "Spain")

    def test_prepare_raises_without_data(self, tmp_path):
        from config.config_loader import load_config

        config = load_config()
        config.international.dataset_path = str(tmp_path / "missing.csv")
        with pytest.raises(ValueError):
            InternationalMatchPredictor(config).prepare()
