"""Exponential time-decay weighting for training samples.

Recent matches carry more predictive signal than old ones. This module
turns match dates into per-sample weights that halve every
``half_life_days``, so the model favours recent form while still learning
from older seasons (which makes deeper history safe to include).
"""

import numpy as np
import pandas as pd

from config.config_loader import TimeDecayConfig


class TimeDecayWeighter:
    """Computes exponential-decay sample weights from match dates.

    ``weight = 0.5 ** (age_days / half_life_days)`` relative to the most
    recent match in the provided dates. When disabled, all weights are
    ``1.0`` (equivalent to no weighting).
    """

    def __init__(self, config: TimeDecayConfig) -> None:
        self._config = config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def compute_weights(self, dates: pd.Series) -> np.ndarray:
        """Return positional sample weights aligned with ``dates``.

        Args:
            dates: Match dates (parseable by ``pandas.to_datetime``), in the
                same row order as the training features.
        """
        n = len(dates)
        # A non-positive half-life has no meaningful decay; fall back to
        # uniform weighting rather than dividing by zero.
        if not self._config.enabled or self._config.half_life_days <= 0:
            return np.ones(n, dtype=float)

        parsed = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
        reference = parsed.max()
        age_series = (reference - parsed).dt.total_seconds() / 86400.0
        age_days = age_series.fillna(age_series.max()).to_numpy(dtype=float)

        weights: np.ndarray = np.power(0.5, age_days / self._config.half_life_days)
        return weights
