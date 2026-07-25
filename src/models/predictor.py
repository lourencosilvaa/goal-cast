from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.blend import OutcomeBlender
from src.models.outcome_model import OutcomeProbabilities
from src.models.poisson.dixon_coles import DixonColesModel


@dataclass
class MatchPrediction:
    """Prediction result for a single match.

    When a Dixon-Coles model is available the 1X2 probabilities are the
    blended ensemble+Poisson result; otherwise they are pure ensemble
    output. The extra markets (over/under, BTTS, scorelines) are produced
    separately by the Dixon-Coles-backed match-stats calculator.
    """

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
    POISSON_FILENAME = "poisson_model.joblib"

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        self.ensemble = joblib.load(path / "ensemble_model.joblib")
        self.scaler: Pipeline = joblib.load(path / "scaler.joblib")
        self.feature_names: list[str] = joblib.load(path / "feature_names.joblib")

        # Optional Dixon-Coles model: when the artifact is present the 1X2
        # probabilities are blended and the extra markets are exposed. When
        # absent the predictor behaves exactly as a pure ensemble.
        self.poisson: DixonColesModel | None = None
        self.blender: OutcomeBlender | None = None
        poisson_path = path / self.POISSON_FILENAME
        if poisson_path.exists():
            self.poisson = joblib.load(poisson_path)
            self.blender = OutcomeBlender(self.poisson.blend_weight)

    def predict(self, features: pd.DataFrame) -> list[MatchPrediction]:
        """Predict outcomes for one or more matches."""
        # Ensure all expected features are present, filling missing ones with 0
        X = features.reindex(columns=self.feature_names, fill_value=0).fillna(0)

        X_scaled = self.scaler.transform(X)
        probas = self.ensemble.predict_proba(X_scaled)

        predictions: list[MatchPrediction] = []
        for idx, (_, row) in enumerate(features.iterrows()):
            proba = probas[idx]
            home_team = str(row.get("HomeTeam", "Unknown"))
            away_team = str(row.get("AwayTeam", "Unknown"))

            away_prob = float(proba[0])
            draw_prob = float(proba[1]) if len(proba) > 1 else 0.0
            home_prob = float(proba[2]) if len(proba) > 2 else 0.0

            if self.poisson is not None and self.blender is not None:
                blended = self.blender.blend(
                    OutcomeProbabilities(home_prob, draw_prob, away_prob),
                    self.poisson.predict_outcome(home_team, away_team),
                )
                home_prob, draw_prob, away_prob = (
                    blended.home_win,
                    blended.draw,
                    blended.away_win,
                )

            ordered = [away_prob, draw_prob, home_prob]
            pred_class = int(np.argmax(ordered))
            confidence = float(np.max(ordered))

            predictions.append(
                MatchPrediction(
                    home_team=home_team,
                    away_team=away_team,
                    home_win_prob=home_prob,
                    draw_prob=draw_prob,
                    away_win_prob=away_prob,
                    predicted_outcome=self.OUTCOME_MAP.get(pred_class, "Unknown"),
                    confidence=confidence,
                )
            )

        return predictions

    def predict_single(self, features: dict[str, float]) -> MatchPrediction:
        """Predict outcome for a single match from a feature dict."""
        df = pd.DataFrame([features])
        return self.predict(df)[0]
