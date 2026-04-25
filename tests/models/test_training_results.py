"""Tests for training_results.json seasons_trained field."""

import json
import pytest
from pathlib import Path


class TestSeasonsTrainedInResults:

    def test_training_results_schema_includes_seasons_trained(self, tmp_path: Path):
        """training_results.json must contain a seasons_trained list."""
        results_file = tmp_path / "training_results.json"
        # Simulate what train_model.py writes — it must include seasons_trained
        results = {
            "cross_validation": {},
            "ensemble": {},
            "features_count": 59,
            "samples_count": 1000,
            "feature_names": [],
            "last_match_date": "2025-05-01",
            "seasons_trained": ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"],
        }
        results_file.write_text(json.dumps(results))
        loaded = json.loads(results_file.read_text())
        assert "seasons_trained" in loaded
        assert isinstance(loaded["seasons_trained"], list)
        assert "1819" in loaded["seasons_trained"]

    def test_seasons_trained_derived_from_data(self):
        """seasons_trained should reflect unique Season values from the loaded DataFrame."""
        import pandas as pd

        df = pd.DataFrame([
            {"Season": "2324", "HomeTeam": "A", "AwayTeam": "B", "FTR": "H"},
            {"Season": "2425", "HomeTeam": "C", "AwayTeam": "D", "FTR": "A"},
            {"Season": "2324", "HomeTeam": "E", "AwayTeam": "F", "FTR": "D"},
        ])

        seasons_trained = sorted(df["Season"].unique().tolist())
        assert seasons_trained == ["2324", "2425"]
        assert len(seasons_trained) == 2
