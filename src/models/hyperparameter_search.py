"""Bounded, leakage-safe XGBoost log-loss hyperparameter search.

The search optimises multiclass log-loss (the metric that matters for a
calibrated prediction/value product) using ``TimeSeriesSplit`` so that no
future match ever leaks into a past validation fold. It is intentionally
report-only: callers inspect the result and decide whether to update the
production configuration, so a search never silently mutates it.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from config.config_loader import XGBSearchConfig


@dataclass
class SearchResult:
    """Outcome of a bounded XGBoost log-loss search."""

    best_params: dict[str, Any]
    best_log_loss: float
    n_candidates: int


class XGBoostLogLossSearcher:
    """Searches XGBoost hyperparameters to minimise time-series log-loss."""

    def __init__(self, config: XGBSearchConfig, random_state: int) -> None:
        self.config = config
        self.random_state = random_state

    def _make_cv(self) -> TimeSeriesSplit:
        """Build the leakage-safe cross-validator (never random KFold)."""
        return TimeSeriesSplit(n_splits=self.config.n_splits)

    def search(self, X: pd.DataFrame, y: pd.Series) -> SearchResult:
        """Run the bounded randomized search and return the best result.

        XGBoost handles NaNs natively and is scale-invariant, so the raw
        feature matrix is searched directly without imputation or scaling.
        """
        estimator = XGBClassifier(
            random_state=self.random_state,
            eval_metric="mlogloss",
        )
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=self.config.param_grid,
            n_iter=self.config.n_iter,
            scoring="neg_log_loss",
            cv=self._make_cv(),
            random_state=self.random_state,
            refit=True,
        )
        search.fit(X, y)

        return SearchResult(
            best_params=dict(search.best_params_),
            best_log_loss=float(-search.best_score_),
            n_candidates=len(search.cv_results_["params"]),
        )
