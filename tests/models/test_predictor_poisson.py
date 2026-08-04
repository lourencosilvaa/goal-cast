"""Tests for MatchPredictor's optional Dixon-Coles 1X2 blending.

When no Poisson artifact is present the predictor behaves exactly as before
(pure ensemble). When one is present, the 1X2 probabilities are the blended
ensemble+Poisson result. Extra markets (over/under, BTTS, scorelines) are
produced separately by the match-stats calculator, not here.
"""

import numpy as np
import pandas as pd
import pytest

from config.config_loader import (
    EnsembleConfig,
    LogisticRegressionConfig,
    ModelConfig,
    PoissonConfig,
    RandomForestConfig,
    XGBoostConfig,
)
from src.models.poisson.dixon_coles import DixonColesModel
from src.models.predictor import MatchPredictor
from src.models.trainer import ModelTrainer


def _model_config() -> ModelConfig:
    return ModelConfig(
        test_size=0.5,
        random_state=42,
        logistic_regression=LogisticRegressionConfig(max_iter=200, C=0.5),
        random_forest=RandomForestConfig(
            n_estimators=10, max_depth=4, min_samples_leaf=2
        ),
        xgboost=XGBoostConfig(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
        ensemble=EnsembleConfig(voting="soft", weights=[1, 1, 2]),
    )


def _train_and_save(tmp_path) -> pd.DataFrame:
    """Train a tiny ensemble, save it, and return a feature frame to predict."""
    n = 40
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "home_a": rng.normal(size=n),
            "home_b": rng.normal(size=n),
        }
    )
    y = pd.Series([0, 1, 2, 0] * (n // 4))
    trainer = ModelTrainer(_model_config())
    trainer.feature_names = list(X.columns)
    trainer.build_ensemble(X, y)
    trainer.save_model(tmp_path)

    features = pd.DataFrame(
        [{"HomeTeam": "Strong", "AwayTeam": "Weak", "home_a": 0.5, "home_b": -0.2}]
    )
    return features


def _synthetic_matches(rounds: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    teams = ["Strong", "Weak"]
    attack = {"Strong": 0.6, "Weak": -0.6}
    defense = {"Strong": -0.5, "Weak": 0.5}
    rows = []
    date = pd.Timestamp("2022-01-01")
    for _ in range(rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lam = np.exp(attack[home] + defense[away] + 0.3)
                mu = np.exp(attack[away] + defense[home])
                rows.append(
                    {
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "FTHG": int(rng.poisson(lam)),
                        "FTAG": int(rng.poisson(mu)),
                        "Date": date,
                    }
                )
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


class TestBackwardCompatible:
    def test_no_poisson_artifact_pure_ensemble(self, tmp_path) -> None:
        features = _train_and_save(tmp_path)
        predictor = MatchPredictor(tmp_path)
        assert predictor.poisson is None
        pred = predictor.predict(features)[0]
        # Still a valid 1X2 distribution, dict shape unchanged.
        total = pred.home_win_prob + pred.draw_prob + pred.away_win_prob
        assert abs(total - 1.0) < 1e-6
        d = pred.to_dict()
        assert set(d) == {
            "home_team",
            "away_team",
            "probabilities",
            "predicted_outcome",
            "confidence",
        }


class TestPoissonBlending:
    def test_blended_1x2_when_artifact_present(self, tmp_path) -> None:
        import joblib

        features = _train_and_save(tmp_path)
        poisson = DixonColesModel(
            PoissonConfig(
                enabled=True, max_goals=8, half_life_days=3650, blend_weight=0.5
            )
        ).fit(_synthetic_matches())
        joblib.dump(poisson, tmp_path / "poisson_model.joblib")

        predictor = MatchPredictor(tmp_path)
        assert predictor.poisson is not None
        pred = predictor.predict(features)[0]

        # 1X2 remains a valid distribution after blending.
        total = pred.home_win_prob + pred.draw_prob + pred.away_win_prob
        assert abs(total - 1.0) < 1e-6

    def test_blend_weight_one_matches_poisson_1x2(self, tmp_path) -> None:
        import joblib

        features = _train_and_save(tmp_path)
        poisson = DixonColesModel(
            PoissonConfig(
                enabled=True, max_goals=8, half_life_days=3650, blend_weight=1.0
            )
        ).fit(_synthetic_matches())
        joblib.dump(poisson, tmp_path / "poisson_model.joblib")

        predictor = MatchPredictor(tmp_path)
        pred = predictor.predict(features)[0]

        dc = poisson.predict_outcome("Strong", "Weak").normalized()
        assert pred.home_win_prob == pytest.approx(dc.home_win, abs=1e-6)
        assert pred.away_win_prob == pytest.approx(dc.away_win, abs=1e-6)
