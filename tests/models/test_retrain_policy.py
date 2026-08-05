"""Phase 1 tests for the model refit policy.

The policy answers a judgement question — "is this much new data worth a
refit?" — separately from the factual question the context answers, "is there
data the current model has not seen?". Splitting the two is what lets the
pipeline publish a fresh dataset snapshot while deliberately holding the model
still.

Every threshold and date here is set explicitly per test; nothing is read from
the shipped config.
"""

import pandas as pd

from config.config_loader import RetrainCheckConfig
from src.models.retrain_policy import (
    RetrainContext,
    RetrainDecision,
    ThresholdRetrainPolicy,
)

_TRAINED = pd.Timestamp("2026-05-16")
_NEWER = pd.Timestamp("2026-08-22")


def _config(*, enabled: bool = True, min_new_matches: int = 800) -> RetrainCheckConfig:
    return RetrainCheckConfig(enabled=enabled, min_new_matches=min_new_matches)


def _context(
    *,
    model_exists: bool = True,
    last_trained_date: pd.Timestamp | None = _TRAINED,
    latest_data_date: pd.Timestamp | None = _NEWER,
    new_match_count: int = 0,
) -> RetrainContext:
    return RetrainContext(
        model_exists=model_exists,
        last_trained_date=last_trained_date,
        latest_data_date=latest_data_date,
        new_match_count=new_match_count,
    )


class TestHasNewData:
    """The factual question: is there data the model's snapshot lacks?"""

    def test_true_when_data_is_newer(self):
        assert _context(latest_data_date=_NEWER).has_new_data is True

    def test_false_when_dates_match(self):
        assert _context(latest_data_date=_TRAINED).has_new_data is False

    def test_true_when_nothing_was_trained_yet(self):
        assert _context(last_trained_date=None).has_new_data is True

    def test_false_when_data_date_is_unknown(self):
        """An unknown data date is not evidence of new data."""
        assert _context(latest_data_date=None).has_new_data is False


class TestThresholdRetrainPolicy:
    """The judgement question: is a refit worth it?"""

    def test_refits_when_no_model_exists(self):
        decision = ThresholdRetrainPolicy(_config()).decide(
            _context(model_exists=False, new_match_count=0)
        )
        assert decision.should_retrain is True

    def test_refits_when_model_has_no_recorded_training_date(self):
        decision = ThresholdRetrainPolicy(_config()).decide(
            _context(last_trained_date=None, new_match_count=0)
        )
        assert decision.should_retrain is True

    def test_refits_when_data_date_is_unknown(self):
        """Unknown provenance is resolved conservatively, as before."""
        decision = ThresholdRetrainPolicy(_config()).decide(
            _context(latest_data_date=None, new_match_count=0)
        )
        assert decision.should_retrain is True

    def test_skips_when_there_is_no_new_data(self):
        decision = ThresholdRetrainPolicy(_config()).decide(
            _context(latest_data_date=_TRAINED, new_match_count=0)
        )
        assert decision.should_retrain is False

    def test_skips_below_the_threshold(self):
        decision = ThresholdRetrainPolicy(_config(min_new_matches=800)).decide(
            _context(new_match_count=190)
        )
        assert decision.should_retrain is False

    def test_reason_names_both_numbers_when_holding(self):
        decision = ThresholdRetrainPolicy(_config(min_new_matches=800)).decide(
            _context(new_match_count=190)
        )
        assert "190" in decision.reason and "800" in decision.reason

    def test_refits_at_the_threshold(self):
        decision = ThresholdRetrainPolicy(_config(min_new_matches=800)).decide(
            _context(new_match_count=800)
        )
        assert decision.should_retrain is True

    def test_refits_above_the_threshold(self):
        decision = ThresholdRetrainPolicy(_config(min_new_matches=800)).decide(
            _context(new_match_count=1200)
        )
        assert decision.should_retrain is True

    def test_disabled_policy_never_refits(self):
        decision = ThresholdRetrainPolicy(_config(enabled=False)).decide(
            _context(new_match_count=5000)
        )
        assert decision.should_retrain is False

    def test_disabled_policy_skips_even_without_a_model(self):
        """`enabled: false` means "only --force", including a cold start."""
        decision = ThresholdRetrainPolicy(_config(enabled=False)).decide(
            _context(model_exists=False, new_match_count=0)
        )
        assert decision.should_retrain is False

    def test_every_decision_carries_a_reason(self):
        policy = ThresholdRetrainPolicy(_config(min_new_matches=800))
        for context in (
            _context(model_exists=False),
            _context(new_match_count=190),
            _context(new_match_count=900),
            _context(latest_data_date=_TRAINED),
        ):
            decision = policy.decide(context)
            assert isinstance(decision, RetrainDecision)
            assert decision.reason


class TestThresholdEdges:
    """Phase 3: boundaries around the configured threshold."""

    def test_zero_threshold_refits_on_any_new_data(self):
        """0 restores the pre-policy behaviour: any new match triggers a refit."""
        decision = ThresholdRetrainPolicy(_config(min_new_matches=0)).decide(
            _context(new_match_count=1)
        )
        assert decision.should_retrain is True

    def test_zero_threshold_still_skips_without_new_data(self):
        decision = ThresholdRetrainPolicy(_config(min_new_matches=0)).decide(
            _context(latest_data_date=_TRAINED, new_match_count=0)
        )
        assert decision.should_retrain is False

    def test_negative_threshold_behaves_like_zero(self):
        decision = ThresholdRetrainPolicy(_config(min_new_matches=-5)).decide(
            _context(new_match_count=1)
        )
        assert decision.should_retrain is True

    def test_one_below_the_threshold_still_holds(self):
        decision = ThresholdRetrainPolicy(_config(min_new_matches=800)).decide(
            _context(new_match_count=799)
        )
        assert decision.should_retrain is False

    def test_a_cold_start_ignores_the_threshold(self):
        """A missing model is a correctness gap, not a judgement call."""
        decision = ThresholdRetrainPolicy(_config(min_new_matches=100000)).decide(
            _context(model_exists=False, new_match_count=1)
        )
        assert decision.should_retrain is True
