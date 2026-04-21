import json
from pathlib import Path

from src.evaluation.evaluation_dataset import EvaluationEntry
from src.models.predictor import MatchPrediction


class EvaluationService:
    """
    Stores pre-match predictions to disk and scores them after results are known.

    Predictions are saved per match_date in JSON files under storage_dir/.
    Call store() before matches are played; call score() after results are in.
    """

    def __init__(self, storage_dir: str) -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _file_for_date(self, match_date: str) -> Path:
        return self._storage_dir / f"{match_date}.json"

    def store(self, match_date: str, prediction: MatchPrediction) -> None:
        path = self._file_for_date(match_date)
        existing: list[dict] = []
        if path.exists():
            existing = json.loads(path.read_text())

        entry = EvaluationEntry(
            match_date=match_date,
            home_team=prediction.home_team,
            away_team=prediction.away_team,
            predicted_outcome=prediction.predicted_outcome,
            home_win_prob=prediction.home_win_prob,
            draw_prob=prediction.draw_prob,
            away_win_prob=prediction.away_win_prob,
            confidence=prediction.confidence,
        )
        existing.append(entry.to_dict())
        path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))

    def load_predictions(self, match_date: str) -> list[EvaluationEntry]:
        path = self._file_for_date(match_date)
        if not path.exists():
            raise FileNotFoundError(f"No predictions stored for date: {match_date}")
        data = json.loads(path.read_text())
        return [EvaluationEntry.from_dict(d) for d in data]

    def score(
        self, match_date: str, actual_results: dict[str, str]
    ) -> dict[str, float | int]:
        """
        Score stored predictions against actual results.

        actual_results format: {"HomeTeam_vs_AwayTeam": "Home Win"|"Draw"|"Away Win"}

        Returns accuracy, brier_score, scored_matches, total_predictions.
        """
        entries = self.load_predictions(match_date)
        correct = 0
        scored = 0
        brier_sum = 0.0

        for entry in entries:
            key = f"{entry.home_team}_vs_{entry.away_team}"
            if key not in actual_results:
                continue
            actual = actual_results[key]
            scored += 1
            if entry.predicted_outcome == actual:
                correct += 1
            outcome_probs = {
                "Home Win": entry.home_win_prob,
                "Draw": entry.draw_prob,
                "Away Win": entry.away_win_prob,
            }
            for outcome, prob in outcome_probs.items():
                indicator = 1.0 if outcome == actual else 0.0
                brier_sum += (prob - indicator) ** 2

        accuracy = correct / scored if scored > 0 else 0.0
        brier_score = brier_sum / scored if scored > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier_score, 4),
            "scored_matches": scored,
            "total_predictions": len(entries),
        }
