from abc import ABC, abstractmethod

import pandas as pd


class AbstractMatchDataLoader(ABC):
    """Interface for a source of historical match data.

    Implementations normalize their source into the canonical schema used
    across the prediction pipeline: ``Date, HomeTeam, AwayTeam, FTHG, FTAG,
    FTR`` (plus any source-specific extras such as ``Neutral``/``Tournament``).
    Keeping loaders behind this interface makes data sources interchangeable.
    """

    @abstractmethod
    def load_all(self) -> pd.DataFrame:
        """Return all available matches in the canonical schema."""
        raise NotImplementedError


class AbstractFeatureEngineer(ABC):
    """Interface for turning canonical match rows into model features.

    Implementations expose a single ``build_all_features`` entry point so the
    feature-generation strategy can be swapped (e.g. club-league vs national
    teams) without changing the training/prediction orchestration.
    """

    @abstractmethod
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return *df* enriched with model-ready feature columns."""
        raise NotImplementedError
