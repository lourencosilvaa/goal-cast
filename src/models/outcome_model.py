"""Shared interface for 1X2 match-outcome models.

Both the calibrated ML ensemble and the Dixon-Coles Poisson model produce a
home/draw/away distribution. Expressing that behind a single small interface
keeps the two model families interchangeable and lets them be blended
without either side knowing the other's internals.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Number of mutually exclusive 1X2 outcomes (home, draw, away).
_NUM_OUTCOMES = 3


@dataclass(frozen=True)
class OutcomeProbabilities:
    """A 1X2 probability triple (home win / draw / away win)."""

    home_win: float
    draw: float
    away_win: float

    def normalized(self) -> "OutcomeProbabilities":
        """Return a copy scaled to sum to 1, or uniform if degenerate."""
        total = self.home_win + self.draw + self.away_win
        if total <= 0:
            uniform = 1.0 / _NUM_OUTCOMES
            return OutcomeProbabilities(uniform, uniform, uniform)
        return OutcomeProbabilities(
            self.home_win / total,
            self.draw / total,
            self.away_win / total,
        )


class OutcomeModel(ABC):
    """Predicts a 1X2 distribution for a fixture identified by team names."""

    @abstractmethod
    def predict_outcome(self, home_team: str, away_team: str) -> OutcomeProbabilities:
        """Return the home/draw/away probabilities for the fixture."""
