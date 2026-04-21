import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.model_cache_manager import ModelCacheManager


class TestModelCacheManagerEdgeCases:

    def test_corrupted_training_results_forces_retrain(self, tmp_path):
        (tmp_path / "ensemble_model.joblib").write_bytes(b"fake")
        (tmp_path / "training_results.json").write_text("{invalid json}")
        manager = ModelCacheManager(models_dir=str(tmp_path))
        assert manager.should_retrain() is True

    def test_training_results_missing_last_match_date_key_forces_retrain(self, tmp_path):
        (tmp_path / "ensemble_model.joblib").write_bytes(b"fake")
        (tmp_path / "training_results.json").write_text(json.dumps({"accuracy": 0.5}))
        manager = ModelCacheManager(models_dir=str(tmp_path))
        assert manager.should_retrain() is True

    def test_latest_date_same_as_trained_date_no_retrain(self, tmp_path):
        results = {"last_match_date": "2025-03-01"}
        (tmp_path / "ensemble_model.joblib").write_bytes(b"fake")
        (tmp_path / "training_results.json").write_text(json.dumps(results))
        manager = ModelCacheManager(models_dir=str(tmp_path))
        assert manager.should_retrain(latest_data_date=pd.Timestamp("2025-03-01")) is False

    def test_latest_date_one_day_later_triggers_retrain(self, tmp_path):
        results = {"last_match_date": "2025-03-01"}
        (tmp_path / "ensemble_model.joblib").write_bytes(b"fake")
        (tmp_path / "training_results.json").write_text(json.dumps(results))
        manager = ModelCacheManager(models_dir=str(tmp_path))
        assert manager.should_retrain(latest_data_date=pd.Timestamp("2025-03-02")) is True

    def test_model_dir_does_not_exist_is_handled(self, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        manager = ModelCacheManager(models_dir=str(nonexistent))
        assert manager.model_exists() is False
        assert manager.should_retrain() is True
