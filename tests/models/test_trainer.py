"""Tests for ModelTrainer, focused on leakage-safe preprocessing.

The imputation of missing feature values MUST be fit on the training
split only — never on the full dataset — so the held-out rows cannot
influence the imputed medians.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config.config_loader import (
    EnsembleConfig,
    LogisticRegressionConfig,
    ModelConfig,
    RandomForestConfig,
    TimeDecayConfig,
    XGBoostConfig,
)
from src.models.time_weighting import TimeDecayWeighter
from src.models.trainer import ModelTrainer


def _make_model_config(test_size: float = 0.2) -> ModelConfig:
    return ModelConfig(
        test_size=test_size,
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


class TestPrepareData:
    def test_preserves_nan_no_global_imputation(self) -> None:
        """prepare_data must NOT impute over the full dataset (leak)."""
        df = pd.DataFrame(
            {
                "home_foo": [1.0, np.nan, 3.0, 4.0],
                "away_bar": [1.0, 2.0, 3.0, 4.0],
                "Result": [0, 1, 2, 0],
            }
        )
        trainer = ModelTrainer(_make_model_config())
        X, y, feature_cols = trainer.prepare_data(df)

        assert "home_foo" in feature_cols
        assert X["home_foo"].isna().any(), "NaN should be left for the pipeline"


class TestPreprocessor:
    def test_is_pipeline_with_imputer_and_scaler(self) -> None:
        trainer = ModelTrainer(_make_model_config())
        pre = trainer._make_preprocessor()
        assert isinstance(pre, Pipeline)
        assert "imputer" in pre.named_steps
        assert "scaler" in pre.named_steps

    def test_imputer_fit_on_train_only(self) -> None:
        """After build_ensemble, imputer median == TRAIN median, not full."""
        # Train half (first 10): median of home_a non-NaN is 2.0.
        # Test half (last 10): value 100.0 — would shift a leaked median.
        home_a = [2.0] * 9 + [np.nan] + [100.0] * 10
        X = pd.DataFrame({"home_a": home_a, "home_b": [1.0] * 20})
        y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 2)

        trainer = ModelTrainer(_make_model_config(test_size=0.5))
        trainer.build_ensemble(X, y)

        imputer = trainer.scaler.named_steps["imputer"]
        assert imputer.statistics_[0] == 2.0  # train median, not the full 100.0


def _synthetic_xy(n: int = 20) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = pd.DataFrame(
        {
            "home_a": [float(i % 5) + 1.0 for i in range(n)],
            "home_b": [float((i * 3) % 7) for i in range(n)],
        }
    )
    y = pd.Series([0, 1, 2, 0, 1] * (n // 5))
    ref = pd.Timestamp("2024-01-01")
    dates = pd.Series([ref + pd.Timedelta(days=i) for i in range(n)])
    return X, y, dates


class TestTimeDecayIntegration:
    def test_build_ensemble_accepts_dates_with_weighter(self) -> None:
        X, y, dates = _synthetic_xy()
        weighter = TimeDecayWeighter(
            TimeDecayConfig(enabled=True, half_life_days=10)
        )
        trainer = ModelTrainer(_make_model_config(test_size=0.5), weighter=weighter)
        result = trainer.build_ensemble(X, y, dates=dates)
        assert "accuracy" in result and "log_loss" in result
        assert trainer.ensemble is not None

    def test_cross_validate_accepts_dates_with_weighter(self) -> None:
        X, y, dates = _synthetic_xy()
        weighter = TimeDecayWeighter(
            TimeDecayConfig(enabled=True, half_life_days=10)
        )
        trainer = ModelTrainer(_make_model_config(), weighter=weighter)
        results = trainer.cross_validate(X, y, n_splits=3, dates=dates)
        assert "XGBoost" in results

    def test_build_ensemble_unchanged_without_weighter(self) -> None:
        X, y, _ = _synthetic_xy()
        trainer = ModelTrainer(_make_model_config(test_size=0.5))
        result = trainer.build_ensemble(X, y)
        assert "accuracy" in result


class TestSaveLoadRoundTrip:
    def test_pipeline_scaler_survives_save_load(self, tmp_path) -> None:
        """The imputer+scaler Pipeline must serialize and reload intact."""
        X, y, _ = _synthetic_xy()
        trainer = ModelTrainer(_make_model_config(test_size=0.5))
        trainer.feature_names = list(X.columns)
        trainer.build_ensemble(X, y)
        trainer.save_model(tmp_path)

        reloaded = ModelTrainer(_make_model_config(test_size=0.5))
        reloaded.load_model(tmp_path)

        assert isinstance(reloaded.scaler, Pipeline)
        assert "imputer" in reloaded.scaler.named_steps
        assert reloaded.feature_names == list(X.columns)
        assert reloaded.ensemble is not None


class TestModelSelection:
    def test_gradient_boosting_excluded(self) -> None:
        """GB is strictly dominated by XGBoost in CV — it must not be built."""
        trainer = ModelTrainer(_make_model_config())
        names = set(trainer._build_models().keys())
        assert "Gradient Boosting" not in names

    def test_core_models_present(self) -> None:
        trainer = ModelTrainer(_make_model_config())
        names = set(trainer._build_models().keys())
        assert {"Logistic Regression", "Random Forest", "XGBoost"} <= names
