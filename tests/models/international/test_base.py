import pandas as pd
import pytest

from src.models.international.base import (
    AbstractFeatureEngineer,
    AbstractMatchDataLoader,
)


class _ConcreteLoader(AbstractMatchDataLoader):
    def load_all(self) -> pd.DataFrame:
        return super().load_all()  # type: ignore[safe-super]


class _ConcreteEngineer(AbstractFeatureEngineer):
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().build_all_features(df)  # type: ignore[safe-super]


class TestAbstractBases:
    def test_loader_super_raises(self):
        with pytest.raises(NotImplementedError):
            _ConcreteLoader().load_all()

    def test_engineer_super_raises(self):
        with pytest.raises(NotImplementedError):
            _ConcreteEngineer().build_all_features(pd.DataFrame())
