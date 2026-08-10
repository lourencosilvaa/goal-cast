"""Re-pricing a match that is already under way.

The stored prediction answers "who wins?" from before kick-off. Replaying it at
minute 80 of a 2-0 is not a live prediction, it is the same pre-match number
with a scoreboard next to it — which is exactly what the "Calcular ao vivo"
button used to do.

What changes a live number is the two things the pre-match model never saw: the
score, and how little time remains to change it. Both enter here.

**The model.** Goals arrive as a Poisson process at the rate the pre-match
model expected (its ``expected_goals``, already stored with every prediction).
The goals *still to come* are then Poisson with that rate scaled by the
fraction of the match left, and the current score is a known head start. At
minute 0 this collapses to the pre-match distribution; at minute 90 it
collapses to the actual result.

**Its limits, which are visible in the output.** The scoring rate is assumed
independent of the score, and it is not — a leading side defends and a chasing
one commits, so comebacks come out slightly under-priced. Red cards, injuries
and substitutions are invisible. Stoppage time is not modelled, and the elapsed
minute itself is often an estimate (see
``src/scrapers/results/elapsed.py``). This is a re-pricing, not a simulation,
and it should be read as one.

Pure standard library on purpose: it runs inside the backend image, whose
dependency group carries no scientific stack, and a Poisson PMF over ten goals
is a factorial and an exponential.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

from src.models.outcome_model import OutcomeProbabilities


@dataclass(frozen=True)
class ExpectedGoals:
    """The pre-match scoring rates, one per side, for a full match."""

    home: float
    away: float


@dataclass(frozen=True)
class LiveState:
    """Where the match stands right now."""

    home_goals: int
    away_goals: int
    #: Minutes played. Clamped here, because upstream it may be an estimate.
    elapsed_minutes: int


@dataclass(frozen=True)
class InPlayForecast:
    """The re-priced match: outcome odds plus what fed them.

    The inputs are returned alongside the answer so a UI can show *why* the
    number moved — "33 minutes left, 1.4 expected goals still to come" is what
    makes a 91% legible.
    """

    outcome: OutcomeProbabilities
    #: Share of the match still to play, in ``[0, 1]``.
    remaining_fraction: float
    #: Goals already scored plus goals still expected, per side.
    expected_home_goals: float
    expected_away_goals: float


class InPlayCalculator:
    """Conditional outcome probabilities for a match in progress."""

    #: Regulation time. Stoppage is not modelled — see the module docstring.
    FULL_TIME: ClassVar[int] = 90
    #: Additional goals per side the distribution is summed over. Ten is far
    #: past the point where Poisson mass is measurable at football rates, and
    #: keeping it finite is what lets this avoid a scientific dependency.
    MAX_GOALS: ClassVar[int] = 10

    def outcome(
        self, expected: ExpectedGoals, state: LiveState
    ) -> OutcomeProbabilities:
        return self.forecast(expected, state).outcome

    def forecast(self, expected: ExpectedGoals, state: LiveState) -> InPlayForecast:
        remaining = self._remaining_fraction(state.elapsed_minutes)
        home_rate = self._rate(expected.home) * remaining
        away_rate = self._rate(expected.away) * remaining

        home_pmf = self._poisson_pmf(home_rate)
        away_pmf = self._poisson_pmf(away_rate)

        # The lead the remaining goals have to overturn. Everything below is a
        # sum over the joint distribution of what is still to come.
        lead = state.home_goals - state.away_goals
        home_win = draw = away_win = 0.0
        for home_more, p_home in enumerate(home_pmf):
            for away_more, p_away in enumerate(away_pmf):
                joint = p_home * p_away
                final = lead + home_more - away_more
                if final > 0:
                    home_win += joint
                elif final == 0:
                    draw += joint
                else:
                    away_win += joint

        return InPlayForecast(
            outcome=OutcomeProbabilities(
                home_win=home_win, draw=draw, away_win=away_win
            ).normalized(),
            remaining_fraction=remaining,
            expected_home_goals=state.home_goals + home_rate,
            expected_away_goals=state.away_goals + away_rate,
        )

    # ── pieces ───────────────────────────────────────────────────────────

    def _remaining_fraction(self, elapsed_minutes: int) -> float:
        """Share of the match still to play.

        Clamped at both ends: the elapsed minute is frequently an estimate and
        can arrive negative (a clock skew before kick-off) or past 90 (a match
        deep into stoppage).
        """
        played = max(0, min(self.FULL_TIME, elapsed_minutes))
        return (self.FULL_TIME - played) / self.FULL_TIME

    @staticmethod
    def _rate(value: float) -> float:
        """A scoring rate can be zero but never negative.

        Expected goals arrive from a stored payload rather than from the model
        in this process, so a malformed one must degrade to "no more goals"
        instead of producing a negative Poisson mean.
        """
        return max(0.0, float(value))

    def _poisson_pmf(self, rate: float) -> list[float]:
        """``P(k goals)`` for ``k`` in ``0..MAX_GOALS``.

        A rate of exactly zero is the degenerate case the formula handles
        correctly — ``e^0 · 0^0 / 0! = 1`` for k=0 — but ``0 ** 0`` is written
        out rather than relied upon, because the certainty it encodes (no more
        goals, so the current score is final) is the whole answer at full time.
        """
        if rate <= 0.0:
            return [1.0] + [0.0] * self.MAX_GOALS
        decay = math.exp(-rate)
        return [decay * rate**k / math.factorial(k) for k in range(self.MAX_GOALS + 1)]
