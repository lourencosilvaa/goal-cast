"""Tests for OutcomeProbabilities and the 1X2 OutcomeBlender."""

import pytest

from src.models.blend import OutcomeBlender
from src.models.outcome_model import OutcomeProbabilities


class TestOutcomeProbabilities:
    def test_normalized_sums_to_one(self) -> None:
        probs = OutcomeProbabilities(2.0, 1.0, 1.0).normalized()
        assert probs.home_win == pytest.approx(0.5)
        assert probs.draw == pytest.approx(0.25)
        assert probs.away_win == pytest.approx(0.25)

    def test_normalized_zero_falls_back_to_uniform(self) -> None:
        probs = OutcomeProbabilities(0.0, 0.0, 0.0).normalized()
        assert probs.home_win == pytest.approx(1 / 3)
        assert probs.draw == pytest.approx(1 / 3)
        assert probs.away_win == pytest.approx(1 / 3)


class TestOutcomeBlender:
    def test_weight_half_averages(self) -> None:
        ens = OutcomeProbabilities(0.5, 0.3, 0.2)
        poi = OutcomeProbabilities(0.3, 0.3, 0.4)
        blended = OutcomeBlender(blend_weight=0.5).blend(ens, poi)
        assert blended.home_win == pytest.approx(0.4)
        assert blended.draw == pytest.approx(0.3)
        assert blended.away_win == pytest.approx(0.3)

    def test_weight_zero_returns_ensemble(self) -> None:
        ens = OutcomeProbabilities(0.5, 0.3, 0.2)
        poi = OutcomeProbabilities(0.1, 0.1, 0.8)
        blended = OutcomeBlender(blend_weight=0.0).blend(ens, poi)
        assert blended.home_win == pytest.approx(0.5)
        assert blended.draw == pytest.approx(0.3)
        assert blended.away_win == pytest.approx(0.2)

    def test_weight_one_returns_poisson(self) -> None:
        ens = OutcomeProbabilities(0.5, 0.3, 0.2)
        poi = OutcomeProbabilities(0.1, 0.1, 0.8)
        blended = OutcomeBlender(blend_weight=1.0).blend(ens, poi)
        assert blended.home_win == pytest.approx(0.1)
        assert blended.draw == pytest.approx(0.1)
        assert blended.away_win == pytest.approx(0.8)

    def test_blend_output_normalized(self) -> None:
        ens = OutcomeProbabilities(0.6, 0.3, 0.1)
        poi = OutcomeProbabilities(0.2, 0.5, 0.3)
        blended = OutcomeBlender(blend_weight=0.4).blend(ens, poi)
        total = blended.home_win + blended.draw + blended.away_win
        assert total == pytest.approx(1.0)
