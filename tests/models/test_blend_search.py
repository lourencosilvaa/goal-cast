"""Tests for the blend-weight log-loss sweep.

Picks the ensemble/Poisson 1X2 mix that minimises held-out log-loss, so the
blend weight is chosen empirically rather than by a default guess.
"""

import numpy as np
import pytest

from src.models.blend_search import BlendWeightSweeper


def _one_hot(y: np.ndarray, confident: float = 0.98) -> np.ndarray:
    """Near-deterministic probabilities matching the labels."""
    n = len(y)
    other = (1.0 - confident) / 2.0
    proba = np.full((n, 3), other)
    proba[np.arange(n), y] = confident
    return proba


class TestBlendWeightSweeper:
    def test_prefers_weight_one_when_poisson_is_perfect(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        poisson = _one_hot(y)  # accurate
        ensemble = np.full((6, 3), 1 / 3)  # uninformative
        result = BlendWeightSweeper([0.0, 0.5, 1.0]).sweep(ensemble, poisson, y)
        assert result.best_weight == 1.0

    def test_prefers_weight_zero_when_ensemble_is_perfect(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        ensemble = _one_hot(y)
        poisson = np.full((6, 3), 1 / 3)
        result = BlendWeightSweeper([0.0, 0.5, 1.0]).sweep(ensemble, poisson, y)
        assert result.best_weight == 0.0

    def test_reports_every_candidate_weight(self) -> None:
        y = np.array([0, 1, 2, 0])
        proba = _one_hot(y)
        weights = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = BlendWeightSweeper(weights).sweep(proba, proba, y)
        assert [w for w, _ in result.per_weight] == weights
        assert all(ll > 0 for _, ll in result.per_weight)

    def test_best_log_loss_is_minimum(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        ensemble = _one_hot(y, confident=0.8)
        poisson = _one_hot(y, confident=0.6)
        result = BlendWeightSweeper([0.0, 0.5, 1.0]).sweep(ensemble, poisson, y)
        assert result.best_log_loss == pytest.approx(
            min(ll for _, ll in result.per_weight)
        )

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            BlendWeightSweeper([0.5]).sweep(
                np.full((3, 3), 1 / 3), np.full((2, 3), 1 / 3), np.array([0, 1, 2])
            )

    def test_empty_weights_raise(self) -> None:
        with pytest.raises(ValueError):
            BlendWeightSweeper([])
