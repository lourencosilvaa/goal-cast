import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.model_cache_manager import ModelCacheManager


class TestModelCacheManager:

    def _make_training_results(self, last_match_date: str) -> dict:
        return {
            "cross_validation": {},
            "ensemble": {"accuracy": 0.53},
            "features_count": 59,
            "samples_count": 10000,
            "feature_names": [],
            "last_match_date": last_match_date,
        }

    def test_should_retrain_when_no_model_exists(self, tmp_path):
        manager = ModelCacheManager(models_dir=str(tmp_path))
        assert manager.should_retrain() is True

    def test_should_retrain_when_training_results_missing(self, tmp_path):
        (tmp_path / "ensemble_model.joblib").write_bytes(b"fake")
        manager = ModelCacheManager(models_dir=str(tmp_path))
        assert manager.should_retrain() is True

    def test_should_retrain_when_new_matches_exist(self, tmp_path):
        results = self._make_training_results("2024-12-01")
        results_path = tmp_path / "training_results.json"
        results_path.write_text(json.dumps(results))
        (tmp_path / "ensemble_model.joblib").write_bytes(b"fake")

        latest_date = pd.Timestamp("2025-01-15")
        manager = ModelCacheManager(models_dir=str(tmp_path))
        assert manager.should_retrain(latest_data_date=latest_date) is True

    def test_should_not_retrain_when_data_is_current(self, tmp_path):
        results = self._make_training_results("2025-01-15")
        results_path = tmp_path / "training_results.json"
        results_path.write_text(json.dumps(results))
        (tmp_path / "ensemble_model.joblib").write_bytes(b"fake")

        latest_date = pd.Timestamp("2025-01-15")
        manager = ModelCacheManager(models_dir=str(tmp_path))
        assert manager.should_retrain(latest_data_date=latest_date) is False

    def test_should_retrain_when_latest_date_is_none_and_no_results(self, tmp_path):
        (tmp_path / "ensemble_model.joblib").write_bytes(b"fake")
        manager = ModelCacheManager(models_dir=str(tmp_path))
        assert manager.should_retrain(latest_data_date=None) is True

    def test_get_last_match_date_returns_none_when_missing(self, tmp_path):
        manager = ModelCacheManager(models_dir=str(tmp_path))
        assert manager.get_last_match_date() is None

    def test_get_last_match_date_returns_timestamp(self, tmp_path):
        results = self._make_training_results("2025-01-10")
        (tmp_path / "training_results.json").write_text(json.dumps(results))
        manager = ModelCacheManager(models_dir=str(tmp_path))
        date = manager.get_last_match_date()
        assert date == pd.Timestamp("2025-01-10")

    def test_model_exists_returns_false_when_no_joblib(self, tmp_path):
        manager = ModelCacheManager(models_dir=str(tmp_path))
        assert manager.model_exists() is False

    def test_model_exists_returns_true_when_joblib_present(self, tmp_path):
        (tmp_path / "ensemble_model.joblib").write_bytes(b"fake")
        manager = ModelCacheManager(models_dir=str(tmp_path))
        assert manager.model_exists() is True
