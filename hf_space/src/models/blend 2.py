"""Blend two 1X2 distributions into one.

The final match probabilities used for value detection are a convex
combination of the calibrated ensemble and the Dixon-Coles Poisson model,
controlled by a single weight so the mix is fully configurable.
"""

from src.models.outcome_model import OutcomeProbabilities


class OutcomeBlender:
    """Convex-combines an ensemble and a Poisson 1X2 distribution.

    ``blend_weight`` is the weight placed on the Poisson model: 0.0 returns
    the ensemble unchanged, 1.0 returns the Poisson distribution.
    """

    def __init__(self, blend_weight: float) -> None:
        self.blend_weight = blend_weight

    def blend(
        self,
        ensemble: OutcomeProbabilities,
        poisson: OutcomeProbabilities,
    ) -> OutcomeProbabilities:
        """Return the weighted, renormalized 1X2 distribution."""
        w = self.blend_weight
        ens = ensemble.normalized()
        poi = poisson.normalized()
        return OutcomeProbabilities(
            (1 - w) * ens.home_win + w * poi.home_win,
            (1 - w) * ens.draw + w * poi.draw,
            (1 - w) * ens.away_win + w * poi.away_win,
        ).normalized()
