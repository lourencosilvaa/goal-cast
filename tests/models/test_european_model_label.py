"""The recorded model must name what actually produced the number.

§14 measured the blend and found Dixon-Coles earns no weight on European ties:
the log-loss profile rises monotonically from ``dixon_coles_weight`` 0.00 to
1.00, and the configured 0.60 was worse than ELO alone by +0.0118 nats with a
95% interval of [+0.0042, +0.0199] over 1,301 held-out matches. The weight is
now 0.00, which makes these predictions ELO and nothing else.

A fixed ``MODEL_NAME`` of ``dixon-coles+elo`` would therefore be a lie — and
the ``model`` field exists precisely to stop a prediction overstating what
produced it. Deriving the label from the weight keeps that true in both
directions: it cannot go stale if the weight is ever changed back, which the
growing corpus gives a real reason to try.
"""

from typing import Any

import pytest

from config.config_loader import EloConfig, EuropeanPredictionConfig, load_config
from src.models.elo import FootballELO
from src.models.european_predictor import (
    EuropeanMatchPredictor,
    PredictionRefused,
    elo_outcome_from_expected,
)
from src.models.outcome_model import OutcomeProbabilities


class SpyDixonColes:
    """Records whether its distribution was ever asked for."""

    def __init__(self, probs: OutcomeProbabilities | None = None) -> None:
        self.asked = 0
        self.known_checks: list[str] = []
        self._probs = probs or OutcomeProbabilities(0.7, 0.2, 0.1)

    def knows(self, team: str) -> bool:
        self.known_checks.append(team)
        return team in {"Benfica", "Arsenal"}

    def predict_outcome(self, home_team: str, away_team: str) -> OutcomeProbabilities:
        self.asked += 1
        return self._probs


class ExplodingElo:
    """Fails if consulted — proves the ELO leg was skipped."""

    config = EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0)

    def get_rating(self, team: str) -> float:  # pragma: no cover - must not run
        raise AssertionError("ELO was consulted when its weight was zero")

    def expected_score(self, a: float, b: float) -> float:  # pragma: no cover
        raise AssertionError("ELO was consulted when its weight was zero")


def _elo(ratings: dict[str, float] | None = None) -> FootballELO:
    elo = FootballELO(EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0))
    for team, rating in (ratings or {"Benfica": 1800.0, "Arsenal": 1750.0}).items():
        elo.ratings[team] = rating
    return elo


def _config(**overrides: Any) -> EuropeanPredictionConfig:
    """Explicit every time — never the shipped defaults (§7.3)."""
    settings: dict[str, Any] = {
        "enabled": True,
        "dixon_coles_weight": 0.0,
        "min_matches_per_team": 10,
        "elo_draw_rate": 0.22,
    }
    settings.update(overrides)
    return EuropeanPredictionConfig(**settings)


def _predictor(
    dixon: Any = None, elo: Any = None, config: Any = None
) -> EuropeanMatchPredictor:
    return EuropeanMatchPredictor(
        dixon_coles=dixon if dixon is not None else SpyDixonColes(),
        elo=elo if elo is not None else _elo(),
        config=config or _config(),
        match_counts={"Benfica": 40, "Arsenal": 40},
    )


# ── the label ────────────────────────────────────────────────────────────


class TestTheLabelTracksTheWeight:
    def test_zero_weight_is_elo_alone(self):
        assert _predictor(config=_config(dixon_coles_weight=0.0)).model_name == "elo"

    def test_full_weight_is_dixon_coles_alone(self):
        predictor = _predictor(config=_config(dixon_coles_weight=1.0))
        assert predictor.model_name == "dixon-coles"

    def test_anything_between_is_a_blend(self):
        predictor = _predictor(config=_config(dixon_coles_weight=0.6))
        assert predictor.model_name == "dixon-coles+elo"

    def test_a_weight_just_above_zero_is_still_a_blend(self):
        """0.05 costs 0.0001 nats and would be a defensible choice — but it is
        a blend, and must not be labelled as pure ELO."""
        predictor = _predictor(config=_config(dixon_coles_weight=0.05))
        assert predictor.model_name == "dixon-coles+elo"

    def test_the_label_reaches_the_prediction(self):
        prediction = _predictor().predict("Benfica", "Arsenal")
        assert prediction.model == "elo"

    def test_the_label_reaches_the_serialised_payload(self):
        """This is what lands in Supabase and, from there, in the UI."""
        prediction = _predictor().predict("Benfica", "Arsenal")
        assert prediction.to_dict()["model"] == "elo"

    def test_a_blended_predictor_serialises_its_own_label(self):
        predictor = _predictor(config=_config(dixon_coles_weight=0.6))
        assert predictor.predict("Benfica", "Arsenal").model == "dixon-coles+elo"


# ── the unused leg ───────────────────────────────────────────────────────


class TestTheUnusedLegIsNotComputed:
    def test_dixon_coles_is_never_asked_at_zero_weight(self):
        spy = SpyDixonColes()
        _predictor(dixon=spy).predict("Benfica", "Arsenal")
        assert spy.asked == 0

    def test_dixon_coles_is_asked_when_it_carries_weight(self):
        spy = SpyDixonColes()
        _predictor(dixon=spy, config=_config(dixon_coles_weight=0.6)).predict(
            "Benfica", "Arsenal"
        )
        assert spy.asked == 1

    def test_elo_is_never_asked_at_full_weight(self):
        _predictor(
            elo=ExplodingElo(), config=_config(dixon_coles_weight=1.0)
        ).predict("Benfica", "Arsenal")

    def test_the_history_gate_still_consults_dixon_coles(self):
        """The gate asks whether a club has history, not whether it is blended.

        Dixon-Coles is still the only thing that can answer that: a team absent
        from ``match_counts`` passes the count check, because ``None`` is not
        below a minimum.
        """
        spy = SpyDixonColes()
        _predictor(dixon=spy).predict("Benfica", "Arsenal")
        assert spy.known_checks == ["Benfica", "Arsenal"]

    def test_an_unknown_club_is_still_refused_at_zero_weight(self):
        with pytest.raises(PredictionRefused, match="no history"):
            _predictor().predict("FC Kairat", "Arsenal")


# ── the numbers ──────────────────────────────────────────────────────────


class TestZeroWeightIsExactlyElo:
    def test_the_probabilities_are_the_elo_leg_untouched(self):
        elo = _elo()
        prediction = _predictor(elo=elo).predict("Benfica", "Arsenal")
        expected = elo_outcome_from_expected(
            elo.expected_score(
                elo.get_rating("Benfica") + elo.config.home_advantage,
                elo.get_rating("Arsenal"),
            ),
            0.22,
        )
        assert prediction.home_win_prob == pytest.approx(expected.home_win)
        assert prediction.draw_prob == pytest.approx(expected.draw)
        assert prediction.away_win_prob == pytest.approx(expected.away_win)

    def test_the_dixon_coles_distribution_cannot_move_the_answer(self):
        """The strongest statement of what weight 0 means."""
        mild = _predictor(dixon=SpyDixonColes(OutcomeProbabilities(0.4, 0.3, 0.3)))
        extreme = _predictor(dixon=SpyDixonColes(OutcomeProbabilities(0.0, 0.0, 1.0)))
        assert mild.predict("Benfica", "Arsenal").home_win_prob == pytest.approx(
            extreme.predict("Benfica", "Arsenal").home_win_prob
        )

    def test_full_weight_is_exactly_the_dixon_coles_distribution(self):
        probs = OutcomeProbabilities(0.55, 0.25, 0.20)
        prediction = _predictor(
            dixon=SpyDixonColes(probs),
            elo=ExplodingElo(),
            config=_config(dixon_coles_weight=1.0),
        ).predict("Benfica", "Arsenal")
        normalized = probs.normalized()
        assert prediction.home_win_prob == pytest.approx(normalized.home_win)

    def test_the_probabilities_still_sum_to_one(self):
        prediction = _predictor().predict("Benfica", "Arsenal")
        total = (
            prediction.home_win_prob + prediction.draw_prob + prediction.away_win_prob
        )
        assert total == pytest.approx(1.0)


# ── the shipped configuration ────────────────────────────────────────────


class TestTheShippedConfigAdoptsTheMeasurement:
    """The measurement is only adopted if it reaches config.yaml."""

    def test_the_weight_is_zero(self):
        config = load_config("config/config.yaml")
        assert config.european.prediction.dixon_coles_weight == 0.0

    def test_the_draw_rate_is_the_measured_one(self):
        config = load_config("config/config.yaml")
        assert config.european.prediction.elo_draw_rate == 0.22

    def test_the_defaults_agree_with_the_file(self):
        """A config missing these keys must not silently get the old guesses."""
        defaults = EuropeanPredictionConfig()
        shipped = load_config("config/config.yaml").european.prediction
        assert defaults.dixon_coles_weight == shipped.dixon_coles_weight
        assert defaults.elo_draw_rate == shipped.elo_draw_rate


class TestBoundariesOfTheWeight:
    """The two short-circuits are boundary conditions, so pin the boundary."""

    def test_a_weight_a_hair_below_one_still_consults_elo(self):
        with pytest.raises(AssertionError, match="ELO was consulted"):
            _predictor(
                elo=ExplodingElo(), config=_config(dixon_coles_weight=0.999)
            ).predict("Benfica", "Arsenal")

    def test_a_weight_a_hair_above_zero_still_consults_dixon_coles(self):
        spy = SpyDixonColes()
        _predictor(dixon=spy, config=_config(dixon_coles_weight=0.001)).predict(
            "Benfica", "Arsenal"
        )
        assert spy.asked == 1

    def test_a_hair_above_zero_barely_moves_the_answer(self):
        """Continuity across the short-circuit: no discontinuity at the edge."""
        pure = _predictor().predict("Benfica", "Arsenal")
        nearly = _predictor(config=_config(dixon_coles_weight=1e-9)).predict(
            "Benfica", "Arsenal"
        )
        assert nearly.home_win_prob == pytest.approx(pure.home_win_prob, abs=1e-6)

    def test_a_refusal_raises_before_either_model_is_asked(self):
        spy = SpyDixonColes()
        with pytest.raises(PredictionRefused):
            _predictor(
                dixon=spy, elo=ExplodingElo(), config=_config(dixon_coles_weight=0.6)
            ).predict("FC Kairat", "Arsenal")
        assert spy.asked == 0

    def test_the_thin_evidence_gate_still_applies_at_zero_weight(self):
        predictor = EuropeanMatchPredictor(
            dixon_coles=SpyDixonColes(),
            elo=_elo(),
            config=_config(min_matches_per_team=10),
            match_counts={"Benfica": 3, "Arsenal": 40},
        )
        with pytest.raises(PredictionRefused, match="only 3 recorded matches"):
            predictor.predict("Benfica", "Arsenal")

    def test_confidence_is_the_winning_probability(self):
        prediction = _predictor().predict("Benfica", "Arsenal")
        assert prediction.confidence == pytest.approx(
            max(
                prediction.home_win_prob,
                prediction.draw_prob,
                prediction.away_win_prob,
            )
        )

    def test_the_stronger_home_side_is_favoured(self):
        """Benfica 1800 at home against Arsenal 1750 — a sanity check that
        dropping Dixon-Coles did not invert the remaining leg."""
        prediction = _predictor().predict("Benfica", "Arsenal")
        assert prediction.home_win_prob > prediction.away_win_prob
        assert prediction.predicted_outcome == "Home Win"
