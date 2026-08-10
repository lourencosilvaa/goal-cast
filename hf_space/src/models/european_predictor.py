"""Predict fixtures whose two teams come from different leagues.

The ensemble is deliberately not used. It is trained only on domestic rows —
European results carry no shot, foul or corner columns, so they never enter its
training matrix — which means it has never seen a cross-league fixture and has
no feature that would tell it one had arrived. Feeding it a Champions League
tie would be extrapolation presented as a prediction.

Dixon-Coles and ELO are different, and only because of the corpus calibration.
Dixon-Coles is fitted on domestic *and* European goals, so its attack and
defence parameters are identified across pools rather than arbitrary. ELO
ratings sit on one scale for the same reason. Before that work neither
statement was true, and neither model could have answered this question.

Every prediction records which model produced it. A Dixon-Coles+ELO number
presented as an ensemble one would overstate what the system knows, and the
difference is invisible downstream unless it is carried explicitly.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from src.models.outcome_model import OutcomeProbabilities


def elo_outcome_from_expected(
    expected_home: float, draw_rate: float
) -> OutcomeProbabilities:
    """Turn an ELO expected score into a 1X2 triple.

    ELO yields an expected *score* — a win probability with draws folded in —
    so the draw has to be carved back out. The share used is taken from the
    configured draw rate rather than derived, because ELO simply does not model
    draws and pretending otherwise would invent precision.

    A module-level function rather than a method because the offline sweep
    needs the identical arithmetic over a whole column of expected scores. Two
    copies of this formula would let the tuned model drift away from the served
    one with nothing downstream able to notice.
    """
    remaining = 1.0 - draw_rate
    return OutcomeProbabilities(
        home_win=expected_home * remaining,
        draw=draw_rate,
        away_win=(1.0 - expected_home) * remaining,
    ).normalized()


class PredictionRefused(RuntimeError):
    """The fixture cannot be predicted, with the reason stated.

    Raised rather than returned as a low-confidence guess. A club from a league
    this project does not track — Kairat, Bodø/Glimt, Sparta Prague — has no
    history to predict from, and inventing a number for it would be the same
    mistake the name resolver refuses to make with spellings.
    """


@dataclass
class EuropeanPrediction:
    """A cross-league prediction, carrying the model that produced it.

    Mirrors :class:`src.models.predictor.MatchPrediction` field for field so
    everything downstream — serialisation, the Supabase payload, the frontend
    types — keeps working, with ``model`` added.
    """

    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_outcome: str
    confidence: float
    model: str

    def to_dict(self) -> dict[str, object]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "probabilities": {
                "home_win": round(self.home_win_prob, 4),
                "draw": round(self.draw_prob, 4),
                "away_win": round(self.away_win_prob, 4),
            },
            "predicted_outcome": self.predicted_outcome,
            "confidence": round(self.confidence, 4),
            "model": self.model,
        }


class EuropeanMatchPredictor:
    """Blends Dixon-Coles and ELO for fixtures that span two leagues.

    Both models are injected already fitted, so this class owns no I/O and is
    testable without artefacts. ``match_counts`` comes in separately because a
    fitted ``DixonColesModel`` records parameters, not how much evidence each
    one rests on — and the caller already holds the frame those counts come
    from.
    """

    #: The three things this predictor can be, depending on the configured
    #: weight. Not a single constant: §14 measured the blend and found
    #: Dixon-Coles earns no weight on European ties, so the shipped weight is
    #: 0.0 and these predictions are ELO alone. A fixed ``dixon-coles+elo``
    #: would name a model that did not produce the number — exactly what the
    #: ``model`` field exists to prevent — and a fixed ``elo`` would become
    #: wrong the moment the weight moved. Deriving it stays true either way.
    ELO_ONLY: ClassVar[str] = "elo"
    DIXON_COLES_ONLY: ClassVar[str] = "dixon-coles"
    BLENDED: ClassVar[str] = "dixon-coles+elo"

    #: Label for the most likely outcome, matching the domestic predictor's
    #: vocabulary so the UI needs no special case.
    HOME_WIN: ClassVar[str] = "Home Win"
    DRAW: ClassVar[str] = "Draw"
    AWAY_WIN: ClassVar[str] = "Away Win"

    def __init__(
        self,
        dixon_coles: Any,
        elo: Any,
        config: Any,
        match_counts: Mapping[str, int] | None = None,
    ) -> None:
        self._dixon_coles = dixon_coles
        self._elo = elo
        self._config = config
        self._match_counts = match_counts or {}

    @property
    def model_name(self) -> str:
        """What actually produced the numbers, given the configured weight."""
        weight = self._config.dixon_coles_weight
        if weight <= 0.0:
            return self.ELO_ONLY
        if weight >= 1.0:
            return self.DIXON_COLES_ONLY
        return self.BLENDED

    # ── gates ────────────────────────────────────────────────────────────

    def refusal_reason(self, home_team: str, away_team: str) -> str | None:
        """Why this fixture cannot be predicted, or ``None`` if it can.

        Offered alongside :meth:`predict` so a caller looping over fixtures can
        ask before committing, instead of using exceptions for control flow.

        Dixon-Coles answers the history question even when it carries no
        weight in the blend, because it is the only thing that can: a club
        absent from ``match_counts`` entirely passes the count check below,
        since ``None`` is not below a minimum. The gate asks whether there is
        evidence for a club, which is a different question from which model
        turns that evidence into a number.
        """
        for team in (home_team, away_team):
            if not self._dixon_coles.knows(team):
                return (
                    f"no history for '{team}' — its league is not tracked, so "
                    f"there is nothing to predict from"
                )
            minimum = self._config.min_matches_per_team
            count = self._match_counts.get(team)
            if count is not None and count < minimum:
                return (
                    f"'{team}' has only {count} recorded matches, below the "
                    f"{minimum} needed for its parameters to mean anything"
                )
        return None

    def can_predict(self, home_team: str, away_team: str) -> bool:
        return self.refusal_reason(home_team, away_team) is None

    # ── prediction ───────────────────────────────────────────────────────

    def predict(self, home_team: str, away_team: str) -> EuropeanPrediction:
        reason = self.refusal_reason(home_team, away_team)
        if reason is not None:
            raise PredictionRefused(reason)

        blended = self._probabilities(home_team, away_team)
        outcome, confidence = self._best(blended)
        return EuropeanPrediction(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=blended.home_win,
            draw_prob=blended.draw,
            away_win_prob=blended.away_win,
            predicted_outcome=outcome,
            confidence=confidence,
            model=self.model_name,
        )

    def _probabilities(self, home_team: str, away_team: str) -> OutcomeProbabilities:
        """The blend, with a leg carrying no weight left uncomputed.

        Not merely an optimisation: at weight 0 a Dixon-Coles failure or an odd
        scoreline distribution can no longer perturb a prediction it
        contributes nothing to.
        """
        weight = self._config.dixon_coles_weight
        if weight <= 0.0:
            return self._elo_probabilities(home_team, away_team)

        goals: OutcomeProbabilities = self._dixon_coles.predict_outcome(
            home_team, away_team
        ).normalized()
        if weight >= 1.0:
            return goals

        ratings = self._elo_probabilities(home_team, away_team)
        return OutcomeProbabilities(
            home_win=weight * goals.home_win + (1 - weight) * ratings.home_win,
            draw=weight * goals.draw + (1 - weight) * ratings.draw,
            away_win=weight * goals.away_win + (1 - weight) * ratings.away_win,
        ).normalized()

    def _elo_probabilities(
        self, home_team: str, away_team: str
    ) -> OutcomeProbabilities:
        """Rating difference turned into a 1X2 triple.

        The carving itself lives in :func:`elo_outcome_from_expected`, shared
        with the offline sweep so the tuned model cannot drift from this one.
        """
        expected_home = self._elo.expected_score(
            self._elo.get_rating(home_team) + self._elo.config.home_advantage,
            self._elo.get_rating(away_team),
        )
        return elo_outcome_from_expected(expected_home, self._config.elo_draw_rate)

    def _best(self, probabilities: OutcomeProbabilities) -> tuple[str, float]:
        options = (
            (self.HOME_WIN, probabilities.home_win),
            (self.DRAW, probabilities.draw),
            (self.AWAY_WIN, probabilities.away_win),
        )
        label, value = max(options, key=lambda pair: pair[1])
        return label, value
