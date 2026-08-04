"""Tests for exponential time-decay sample weighting."""

import numpy as np
import pandas as pd

from config.config_loader import TimeDecayConfig
from src.models.time_weighting import TimeDecayWeighter


def _dates(*days_ago: int, reference: str = "2024-01-31") -> pd.Series:
    ref = pd.Timestamp(reference)
    return pd.Series([ref - pd.Timedelta(days=d) for d in days_ago])


class TestTimeDecayWeighter:
    def test_disabled_returns_uniform_weights(self) -> None:
        weighter = TimeDecayWeighter(
            TimeDecayConfig(enabled=False, half_life_days=180)
        )
        weights = weighter.compute_weights(_dates(0, 100, 1000))
        assert np.allclose(weights, 1.0)

    def test_most_recent_match_weight_is_one(self) -> None:
        weighter = TimeDecayWeighter(
            TimeDecayConfig(enabled=True, half_life_days=180)
        )
        weights = weighter.compute_weights(_dates(0, 180, 360))
        assert weights[0] == 1.0

    def test_half_life_halves_the_weight(self) -> None:
        weighter = TimeDecayWeighter(
            TimeDecayConfig(enabled=True, half_life_days=180)
        )
        weights = weighter.compute_weights(_dates(0, 180, 360))
        assert abs(weights[1] - 0.5) < 1e-9
        assert abs(weights[2] - 0.25) < 1e-9

    def test_weights_are_monotonically_decreasing_with_age(self) -> None:
        weighter = TimeDecayWeighter(
            TimeDecayConfig(enabled=True, half_life_days=90)
        )
        weights = weighter.compute_weights(_dates(0, 30, 90, 365))
        assert list(weights) == sorted(weights, reverse=True)

    def test_length_matches_input(self) -> None:
        weighter = TimeDecayWeighter(
            TimeDecayConfig(enabled=True, half_life_days=90)
        )
        weights = weighter.compute_weights(_dates(0, 10, 20, 30, 40))
        assert len(weights) == 5

    def test_zero_half_life_falls_back_to_uniform(self) -> None:
        weighter = TimeDecayWeighter(
            TimeDecayConfig(enabled=True, half_life_days=0)
        )
        weights = weighter.compute_weights(_dates(0, 100, 1000))
        assert np.allclose(weights, 1.0)

    def test_very_large_half_life_is_near_uniform(self) -> None:
        weighter = TimeDecayWeighter(
            TimeDecayConfig(enabled=True, half_life_days=1_000_000)
        )
        weights = weighter.compute_weights(_dates(0, 365, 3650))
        assert np.allclose(weights, 1.0, atol=1e-2)

    def test_empty_dates_returns_empty(self) -> None:
        weighter = TimeDecayWeighter(
            TimeDecayConfig(enabled=True, half_life_days=90)
        )
        weights = weighter.compute_weights(pd.Series([], dtype="datetime64[ns]"))
        assert len(weights) == 0

    def test_unparseable_dates_get_max_age(self) -> None:
        weighter = TimeDecayWeighter(
            TimeDecayConfig(enabled=True, half_life_days=90)
        )
        dates = pd.Series(["2024-01-31", "not-a-date", "2023-01-31"])
        weights = weighter.compute_weights(dates)
        assert len(weights) == 3
        # The unparseable entry is treated as oldest → smallest weight.
        assert weights[1] == weights.min()
