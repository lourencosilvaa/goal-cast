from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class MatchPrediction:
    """Prediction result for a single match."""

    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_outcome: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        """Convert prediction to dictionary."""
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "probabilities": {
                "home_win": round(self.home_win_prob, 4),
                "draw": round(self.draw_prob, 4),
                "away_win": round(self.away_win_prob, 4),
            },
            "predicted_outcome": self.predicted_outcome,
            "confidence": round(self.confidence, 4),
        }


class MatchPredictor:
    """Predicts match outcomes using trained ensemble model."""

    OUTCOME_MAP = {0: "Away Win", 1: "Draw", 2: "Home Win"}

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        self.ensemble = joblib.load(path / "ensemble_model.joblib")
        self.scaler: StandardScaler = joblib.load(path / "scaler.joblib")
        self.feature_names: list[str] = joblib.load(path / "feature_names.joblib")

    def predict(self, features: pd.DataFrame) -> list[MatchPrediction]:
        """Predict outcomes for one or more matches."""
        available = [c for c in self.feature_names if c in features.columns]
        X = features[available].fillna(0)

        X_scaled = self.scaler.transform(X)
        probas = self.ensemble.predict_proba(X_scaled)

        predictions: list[MatchPrediction] = []
        for idx, (_, row) in enumerate(features.iterrows()):
            proba = probas[idx]
            pred_class = int(np.argmax(proba))
            confidence = float(np.max(proba))

            predictions.append(
                MatchPrediction(
                    home_team=str(row.get("HomeTeam", "Unknown")),
                    away_team=str(row.get("AwayTeam", "Unknown")),
                    home_win_prob=float(proba[2]) if len(proba) > 2 else 0.0,
                    draw_prob=float(proba[1]) if len(proba) > 1 else 0.0,
                    away_win_prob=float(proba[0]),
                    predicted_outcome=self.OUTCOME_MAP.get(pred_class, "Unknown"),
                    confidence=confidence,
                )
            )

        return predictions

    def predict_single(self, features: dict[str, float]) -> MatchPrediction:
        """Predict outcome for a single match from a feature dict."""
        df = pd.DataFrame([features])
        return self.predict(df)[0]
