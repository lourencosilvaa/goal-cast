import json
from pathlib import Path

import pandas as pd


class ModelCacheManager:
    """
    Determines whether the ML model needs retraining by comparing
    the date of the latest match in available data against the date
    of the last match used when training the saved model.
    """

    _RESULTS_FILE = "training_results.json"
    _MODEL_FILE = "ensemble_model.joblib"

    def __init__(self, models_dir: str) -> None:
        self._models_dir = Path(models_dir)

    def model_exists(self) -> bool:
        return (self._models_dir / self._MODEL_FILE).exists()

    def get_last_match_date(self) -> pd.Timestamp | None:
        results_path = self._models_dir / self._RESULTS_FILE
        if not results_path.exists():
            return None
        try:
            data = json.loads(results_path.read_text())
            raw = data.get("last_match_date")
            if raw:
                return pd.Timestamp(raw)
        except Exception:
            pass
        return None

    def should_retrain(self, latest_data_date: pd.Timestamp | None = None) -> bool:
        """
        Returns True when retraining is needed:
        - No saved model exists, or
        - training_results.json is missing or has no last_match_date, or
        - latest_data_date is later than the stored last_match_date.
        """
        if not self.model_exists():
            return True
        last_trained = self.get_last_match_date()
        if last_trained is None:
            return True
        if latest_data_date is None:
            return True
        return latest_data_date > last_trained
