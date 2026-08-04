"""Empirical selection of the ensemble/Poisson blend weight.

Given held-out ensemble and Dixon-Coles 1X2 probabilities and the true
outcomes, this sweeps a grid of blend weights and returns the one that
minimises multiclass log-loss — the metric that matters for a calibrated
prediction/value product. Report-only: it never mutates configuration.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss

# The 1X2 class labels, ordered to match the probability columns
# (away=0, draw=1, home=2) used throughout the predictor.
_LABELS = [0, 1, 2]


@dataclass
class BlendSweepResult:
    """Outcome of a blend-weight sweep."""

    best_weight: float
    best_log_loss: float
    per_weight: list[tuple[float, float]]


class BlendWeightSweeper:
    """Sweeps blend weights to minimise held-out log-loss.

    ``weights`` is the grid of Poisson weights to try (0 = ensemble only,
    1 = Poisson only), mirroring ``OutcomeBlender`` semantics.
    """

    def __init__(self, weights: list[float]) -> None:
        if not weights:
            raise ValueError("weights must be a non-empty grid")
        self.weights = weights

    def sweep(
        self,
        ensemble_proba: np.ndarray,
        poisson_proba: np.ndarray,
        y_true: np.ndarray,
    ) -> BlendSweepResult:
        """Return the best weight and per-weight log-loss.

        ``ensemble_proba`` and ``poisson_proba`` are (n, 3) arrays whose
        columns are ordered [away, draw, home]; ``y_true`` holds the encoded
        outcomes (0/1/2).
        """
        if not (len(ensemble_proba) == len(poisson_proba) == len(y_true)):
            raise ValueError(
                "ensemble_proba, poisson_proba and y_true must align in length"
            )

        per_weight: list[tuple[float, float]] = []
        for w in self.weights:
            blended = (1.0 - w) * ensemble_proba + w * poisson_proba
            # Both inputs sum to 1, so the mix does too; renormalize
            # defensively against floating-point drift.
            blended = blended / blended.sum(axis=1, keepdims=True)
            loss = float(log_loss(y_true, blended, labels=_LABELS))
            per_weight.append((w, loss))

        best_weight, best_log_loss = min(per_weight, key=lambda item: item[1])
        return BlendSweepResult(
            best_weight=best_weight,
            best_log_loss=best_log_loss,
            per_weight=per_weight,
        )
