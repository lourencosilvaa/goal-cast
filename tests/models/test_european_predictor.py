"""Tests for predicting cross-league fixtures.

The ensemble deliberately has no part in this. It is trained only on domestic
rows — European results carry no shot, foul or corner columns, so they never
enter its training matrix — which means it has never seen a fixture whose two
teams come from different leagues and has no feature that would tell it one
had arrived. Asking it anyway would be extrapolation dressed up as a
prediction.

Dixon-Coles and ELO are different. Both were made comparable across leagues by
the Phase 3 calibration: Dixon-Coles is fitted on ``combined_goals()``, so its
attack and defence parameters are identified across pools, and ELO ratings now
sit on one scale. Those two are what predict here, and every prediction says so.

The other half of the contract is refusal. A club from a league this project
does not track has no history to predict from, and saying so is the only
honest answer.
"""

import pytest

from config.config_loader import EloConfig, EuropeanPredictionConfig
from src.models.elo import FootballELO
from src.models.european_predictor import (
    EuropeanMatchPredictor,
    PredictionRefused,
)
from src.models.outcome_model import OutcomeProbabilities


class _DixonColes:
    """Stands in for a fitted DixonColesModel.

    Mirrors the real interface, which offers ``knows`` and ``predict_outcome``
    but no notion of how much history a team has. Match counts are therefore
    injected into the predictor separately rather than bolted onto the fitted
    model — the caller already holds the frame they come from.
    """

    def __init__(self, known: dict[str, int] | None = None, probs=None) -> None:
        self._known = known if known is not None else {"Benfica": 40, "Arsenal": 40}
        self._probs = probs or OutcomeProbabilities(0.5, 0.25, 0.25)

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._known)

    def knows(self, team: str) -> bool:
        return team in self._known

    def predict_outcome(self, home_team: str, away_team: str):
        return self._probs


def _elo(ratings: dict[str, float] | None = None) -> FootballELO:
    elo = FootballELO(
        EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0)
    )
    for team, rating in (ratings or {"Benfica": 1800.0, "Arsenal": 1750.0}).items():
        elo.ratings[team] = rating
    return elo


def _config(**overrides) -> EuropeanPredictionConfig:
    base = dict(enabled=True, dixon_coles_weight=0.6, min_matches_per_team=10)
    base.update(overrides)
    return EuropeanPredictionConfig(**base)


def _predictor(dixon=None, elo=None, config=None) -> EuropeanMatchPredictor:
    model = dixon or _DixonColes()
    return EuropeanMatchPredictor(
        dixon_coles=model,
        elo=elo or _elo(),
        config=config or _config(),
        match_counts=model.counts,
    )


class TestPrediction:
    def test_a_known_pairing_predicts(self):
        prediction = _predictor().predict("Benfica", "Arsenal")
        assert prediction.home_team == "Benfica"
        assert prediction.away_team == "Arsenal"

    def test_probabilities_sum_to_one(self):
        p = _predictor().predict("Benfica", "Arsenal")
        assert p.home_win_prob + p.draw_prob + p.away_win_prob == pytest.approx(1.0)

    def test_every_probability_is_a_valid_probability(self):
        p = _predictor().predict("Benfica", "Arsenal")
        for value in (p.home_win_prob, p.draw_prob, p.away_win_prob):
            assert 0.0 <= value <= 1.0

    def test_the_predicted_outcome_is_the_most_likely_one(self):
        dixon = _DixonColes(probs=OutcomeProbabilities(0.1, 0.1, 0.8))
        p = _predictor(dixon=dixon, elo=_elo({"Benfica": 1500.0, "Arsenal": 1900.0}))
        assert p.predict("Benfica", "Arsenal").predicted_outcome == "Away Win"

    def test_confidence_is_the_winning_probability(self):
        p = _predictor().predict("Benfica", "Arsenal")
        assert p.confidence == pytest.approx(
            max(p.home_win_prob, p.draw_prob, p.away_win_prob)
        )


class TestModelLabelling:
    """A Dixon-Coles+ELO number must never be presented as an ensemble one."""

    def test_the_prediction_names_its_model(self):
        p = _predictor().predict("Benfica", "Arsenal")
        assert p.model == _predictor().model_name

    def test_the_label_is_not_the_domestic_one(self):
        p = _predictor().predict("Benfica", "Arsenal")
        assert "ensemble" not in p.model.lower()

    def test_the_label_survives_serialisation(self):
        p = _predictor().predict("Benfica", "Arsenal")
        assert p.to_dict()["model"] == _predictor().model_name


class TestBlending:
    def test_full_weight_uses_dixon_coles_alone(self):
        dixon = _DixonColes(probs=OutcomeProbabilities(0.7, 0.2, 0.1))
        p = _predictor(dixon=dixon, config=_config(dixon_coles_weight=1.0)).predict(
            "Benfica", "Arsenal"
        )
        assert p.home_win_prob == pytest.approx(0.7)

    def test_zero_weight_uses_elo_alone(self):
        """Equal ratings plus home advantage must still favour the home side."""
        dixon = _DixonColes(probs=OutcomeProbabilities(0.0, 0.0, 1.0))
        p = _predictor(
            elo=_elo({"Benfica": 1600.0, "Arsenal": 1600.0}),
            config=_config(dixon_coles_weight=0.0),
        ).predict("Benfica", "Arsenal")
        assert p.home_win_prob > p.away_win_prob

    def test_the_blend_sits_between_its_two_inputs(self):
        dixon = _DixonColes(probs=OutcomeProbabilities(1.0, 0.0, 0.0))
        blended = _predictor(dixon=dixon, config=_config(dixon_coles_weight=0.5))
        pure_dc = _predictor(dixon=dixon, config=_config(dixon_coles_weight=1.0))
        pure_elo = _predictor(dixon=dixon, config=_config(dixon_coles_weight=0.0))
        low = min(
            pure_dc.predict("Benfica", "Arsenal").home_win_prob,
            pure_elo.predict("Benfica", "Arsenal").home_win_prob,
        )
        high = max(
            pure_dc.predict("Benfica", "Arsenal").home_win_prob,
            pure_elo.predict("Benfica", "Arsenal").home_win_prob,
        )
        assert low <= blended.predict("Benfica", "Arsenal").home_win_prob <= high


class TestRefusal:
    def test_an_unknown_home_team_is_refused(self):
        """Kairat plays in a league this project does not track."""
        with pytest.raises(PredictionRefused):
            _predictor().predict("FC Kairat", "Arsenal")

    def test_an_unknown_away_team_is_refused(self):
        with pytest.raises(PredictionRefused):
            _predictor().predict("Arsenal", "FC Kairat")

    def test_the_refusal_names_the_team(self):
        with pytest.raises(PredictionRefused, match="Kairat"):
            _predictor().predict("FC Kairat", "Arsenal")

    def test_too_little_history_is_refused(self):
        """A rating from three matches is noise, not knowledge."""
        dixon = _DixonColes(known={"Benfica": 40, "Arsenal": 3})
        with pytest.raises(PredictionRefused, match="Arsenal"):
            _predictor(dixon=dixon, config=_config(min_matches_per_team=10)).predict(
                "Benfica", "Arsenal"
            )

    def test_exactly_the_minimum_is_accepted(self):
        dixon = _DixonColes(known={"Benfica": 40, "Arsenal": 10})
        predictor = _predictor(dixon=dixon, config=_config(min_matches_per_team=10))
        assert predictor.predict("Benfica", "Arsenal")

    def test_can_predict_reports_without_raising(self):
        """Callers looping over fixtures need to ask before committing."""
        predictor = _predictor()
        assert predictor.can_predict("Benfica", "Arsenal")
        assert not predictor.can_predict("FC Kairat", "Arsenal")

    def test_the_reason_is_available_without_catching(self):
        assert "Kairat" in (_predictor().refusal_reason("FC Kairat", "Arsenal") or "")

    def test_no_reason_when_the_pairing_is_predictable(self):
        assert _predictor().refusal_reason("Benfica", "Arsenal") is None
