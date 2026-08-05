"""When is new match data worth refitting the model?

Two questions hide behind "should we retrain", and conflating them is what
freezes a deployment. The first is factual: is there match data the current
model's snapshot does not contain? That governs whether a fresh dataset should
be published and the inference service restarted, and it is answered by
:attr:`RetrainContext.has_new_data`. The second is a judgement call: is the
new data *worth* a refit? Time-decay weighting means a single round of matches
moves a multi-season model barely at all, so refitting on it costs a full
pipeline run, an artefact upload and a service restart to change probabilities
in the third decimal.

:class:`RetrainPolicy` owns only the second question. Keeping it behind an
interface leaves room for a season-aware or calendar-based rule later without
touching the training script.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config.config_loader import RetrainCheckConfig


@dataclass(frozen=True)
class RetrainContext:
    """Everything a policy needs to judge a refit, as one object."""

    #: Whether a trained artefact exists on disk at all.
    model_exists: bool
    #: Date of the newest match the saved model was trained on, if recorded.
    last_trained_date: Optional[pd.Timestamp]
    #: Date of the newest match in the freshly loaded data, if determinable.
    latest_data_date: Optional[pd.Timestamp]
    #: Matches newer than ``last_trained_date`` in the freshly loaded data.
    new_match_count: int

    @property
    def has_new_data(self) -> bool:
        """Whether the loaded data holds matches the model has not seen.

        An undeterminable data date is not evidence of new data — it is
        evidence of nothing, and must not trigger a republish.
        """
        if self.latest_data_date is None:
            return False
        if self.last_trained_date is None:
            return True
        return bool(self.latest_data_date > self.last_trained_date)


@dataclass(frozen=True)
class RetrainDecision:
    """A refit verdict plus the reason, so CI logs say why it went that way."""

    should_retrain: bool
    reason: str


class RetrainPolicy(ABC):
    """Decides whether the accumulated new data justifies a refit."""

    @abstractmethod
    def decide(self, context: RetrainContext) -> RetrainDecision:
        """Judge one refit opportunity."""


class ThresholdRetrainPolicy(RetrainPolicy):
    """Refits once enough new matches have accumulated.

    A missing or unattributable model is always rebuilt — those are correctness
    gaps, not judgement calls. Beyond that the rule is a single number:
    ``min_new_matches``. Set it to 0 to refit on any new data.
    """

    def __init__(self, config: RetrainCheckConfig) -> None:
        self._config = config

    def decide(self, context: RetrainContext) -> RetrainDecision:
        if not self._config.enabled:
            return RetrainDecision(False, "automatic retraining is disabled")

        if not context.model_exists:
            return RetrainDecision(True, "no trained model exists yet")

        if context.last_trained_date is None:
            return RetrainDecision(True, "model has no recorded training date")

        if context.latest_data_date is None:
            return RetrainDecision(True, "latest data date could not be determined")

        if not context.has_new_data:
            return RetrainDecision(
                False,
                f"no matches newer than {context.last_trained_date.date()}",
            )

        threshold = self._config.min_new_matches
        if context.new_match_count < threshold:
            return RetrainDecision(
                False,
                f"only {context.new_match_count} new matches, "
                f"below the {threshold} needed to refit",
            )

        return RetrainDecision(
            True, f"{context.new_match_count} new matches (threshold {threshold})"
        )
