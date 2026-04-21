import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.evaluation_dataset import EvaluationEntry
from src.evaluation.evaluation_service import EvaluationService
from src.models.predictor import MatchPrediction


def _make_prediction(home_team: str = "Arsenal", away_team: str = "Chelsea") -> MatchPrediction:
    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        home_win_prob=0.55,
        draw_prob=0.25,
        away_win_prob=0.20,
        predicted_outcome="Home Win",
        confidence=0.55,
    )


class TestEvaluationService:

    def _make_service(self, tmp_path: Path) -> EvaluationService:
        return EvaluationService(storage_dir=str(tmp_path))

    def test_store_creates_json_file(self, tmp_path):
        service = self._make_service(tmp_path)
        pred = _make_prediction()
        service.store(match_date="2026-04-20", prediction=pred)

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1

    def test_store_persists_correct_fields(self, tmp_path):
        service = self._make_service(tmp_path)
        pred = _make_prediction()
        service.store(match_date="2026-04-20", prediction=pred)

        json_files = list(tmp_path.glob("*.json"))
        data = json.loads(json_files[0].read_text())
        assert isinstance(data, list)
        entry = data[0]
        assert entry["home_team"] == "Arsenal"
        assert entry["away_team"] == "Chelsea"
        assert entry["predicted_outcome"] == "Home Win"
        assert "home_win_prob" in entry
        assert "match_date" in entry

    def test_store_multiple_predictions_same_date(self, tmp_path):
        service = self._make_service(tmp_path)
        service.store("2026-04-20", _make_prediction("Arsenal", "Chelsea"))
        service.store("2026-04-20", _make_prediction("Man City", "Liverpool"))

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert len(data) == 2

    def test_score_returns_accuracy_metric(self, tmp_path):
        service = self._make_service(tmp_path)
        service.store("2026-04-20", _make_prediction("Arsenal", "Chelsea"))

        results = {"Arsenal_vs_Chelsea": "Home Win"}
        metrics = service.score(match_date="2026-04-20", actual_results=results)

        assert "accuracy" in metrics
        assert metrics["accuracy"] == 1.0

    def test_score_returns_zero_accuracy_on_wrong_prediction(self, tmp_path):
        service = self._make_service(tmp_path)
        service.store("2026-04-20", _make_prediction("Arsenal", "Chelsea"))

        results = {"Arsenal_vs_Chelsea": "Away Win"}
        metrics = service.score(match_date="2026-04-20", actual_results=results)

        assert metrics["accuracy"] == 0.0

    def test_score_handles_partial_results(self, tmp_path):
        service = self._make_service(tmp_path)
        service.store("2026-04-20", _make_prediction("Arsenal", "Chelsea"))
        service.store("2026-04-20", _make_prediction("Man City", "Liverpool"))

        results = {"Arsenal_vs_Chelsea": "Home Win"}
        metrics = service.score(match_date="2026-04-20", actual_results=results)

        assert metrics["accuracy"] == 1.0
        assert metrics["scored_matches"] == 1
        assert metrics["total_predictions"] == 2

    def test_score_raises_when_no_predictions_for_date(self, tmp_path):
        service = self._make_service(tmp_path)
        with pytest.raises(FileNotFoundError):
            service.score(match_date="2026-04-20", actual_results={})

    def test_load_predictions_for_date(self, tmp_path):
        service = self._make_service(tmp_path)
        service.store("2026-04-20", _make_prediction())
        entries = service.load_predictions(match_date="2026-04-20")
        assert len(entries) == 1
        assert isinstance(entries[0], EvaluationEntry)

    def test_evaluation_entry_dataclass_has_required_fields(self):
        entry = EvaluationEntry(
            match_date="2026-04-20",
            home_team="Arsenal",
            away_team="Chelsea",
            predicted_outcome="Home Win",
            home_win_prob=0.55,
            draw_prob=0.25,
            away_win_prob=0.20,
            confidence=0.55,
        )
        assert entry.match_date == "2026-04-20"
        assert entry.predicted_outcome == "Home Win"
