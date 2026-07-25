"""Tests for the bounded, leakage-safe XGBoost log-loss search.

The search MUST use TimeSeriesSplit (never random CV) so no future data
leaks into past folds, and MUST stay bounded by the configured n_iter.
"""

import numpy as np
import pandas as pd

from config.config_loader import XGBSearchConfig
from src.models.hyperparameter_search import XGBoostLogLossSearcher


def _make_search_config(n_iter: int = 4, n_splits: int = 3) -> XGBSearchConfig:
    return XGBSearchConfig(
        enabled=True,
        n_iter=n_iter,
        n_splits=n_splits,
        param_grid={
            "n_estimators": [10, 20],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    )


def _xy(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    signal = rng.normal(size=n)
    y = pd.Series(np.clip(((signal + 1.5) // 1).astype(int), 0, 2))
    X = pd.DataFrame(
        {
            "home_a": signal + rng.normal(scale=0.3, size=n),
            "home_b": rng.normal(size=n),
        }
    )
    return X, y


class TestXGBoostLogLossSearcher:
    def test_returns_best_params_and_log_loss(self) -> None:
        X, y = _xy()
        searcher = XGBoostLogLossSearcher(_make_search_config(), random_state=42)
        result = searcher.search(X, y)

        assert set(result.best_params).issuperset(
            {"n_estimators", "max_depth", "learning_rate"}
        )
        assert result.best_log_loss > 0.0

    def test_respects_n_iter_bound(self) -> None:
        X, y = _xy()
        searcher = XGBoostLogLossSearcher(
            _make_search_config(n_iter=2), random_state=42
        )
        result = searcher.search(X, y)
        assert result.n_candidates == 2

    def test_uses_time_series_split(self) -> None:
        from sklearn.model_selection import TimeSeriesSplit

        searcher = XGBoostLogLossSearcher(_make_search_config(), random_state=42)
        assert isinstance(searcher._make_cv(), TimeSeriesSplit)

    def test_single_candidate_search(self) -> None:
        X, y = _xy()
        searcher = XGBoostLogLossSearcher(
            _make_search_config(n_iter=1, n_splits=3), random_state=7
        )
        result = searcher.search(X, y)
        assert result.n_candidates == 1
        assert "n_estimators" in result.best_params

    def test_integer_params_stay_integers(self) -> None:
        """n_estimators/max_depth must reach XGBoost as ints, not floats."""
        X, y = _xy()
        searcher = XGBoostLogLossSearcher(_make_search_config(), random_state=42)
        result = searcher.search(X, y)
        assert isinstance(result.best_params["n_estimators"], int)
        assert isinstance(result.best_params["max_depth"], int)
