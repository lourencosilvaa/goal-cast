import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.evaluation_service import EvaluationService
from src.models.predictor import MatchPrediction


def _make_pred(home: str = "Arsenal", away: str = "Chelsea", outcome: str = "Home Win") -> MatchPrediction:
    return MatchPrediction(
        home_team=home,
        away_team=away,
        home_win_prob=0.55,
        draw_prob=0.25,
        away_win_prob=0.20,
        predicted_outcome=outcome,
        confidence=0.55,
    )


class TestEvaluationServiceEdgeCases:

    def test_score_with_no_matching_results_returns_zero_scored(self, tmp_path):
        service = EvaluationService(storage_dir=str(tmp_path))
        service.store("2026-04-20", _make_pred())
        metrics = service.score("2026-04-20", actual_results={"Unknown_vs_Unknown": "Home Win"})
        assert metrics["scored_matches"] == 0
        assert metrics["accuracy"] == 0.0

    def test_score_all_correct_returns_accuracy_one(self, tmp_path):
        service = EvaluationService(storage_dir=str(tmp_path))
        service.store("2026-04-20", _make_pred("A", "B", "Home Win"))
        service.store("2026-04-20", _make_pred("C", "D", "Draw"))
        results = {"A_vs_B": "Home Win", "C_vs_D": "Draw"}
        metrics = service.score("2026-04-20", results)
        assert metrics["accuracy"] == 1.0

    def test_score_all_wrong_returns_accuracy_zero(self, tmp_path):
        service = EvaluationService(storage_dir=str(tmp_path))
        service.store("2026-04-20", _make_pred("A", "B", "Home Win"))
        metrics = service.score("2026-04-20", {"A_vs_B": "Away Win"})
        assert metrics["accuracy"] == 0.0

    def test_brier_score_is_returned(self, tmp_path):
        service = EvaluationService(storage_dir=str(tmp_path))
        service.store("2026-04-20", _make_pred())
        metrics = service.score("2026-04-20", {"Arsenal_vs_Chelsea": "Home Win"})
        assert "brier_score" in metrics
        assert isinstance(metrics["brier_score"], float)

    def test_store_across_different_dates_creates_separate_files(self, tmp_path):
        service = EvaluationService(storage_dir=str(tmp_path))
        service.store("2026-04-20", _make_pred())
        service.store("2026-04-21", _make_pred("Man City", "Liverpool"))
        assert (tmp_path / "2026-04-20.json").exists()
        assert (tmp_path / "2026-04-21.json").exists()

    def test_load_predictions_raises_for_nonexistent_date(self, tmp_path):
        service = EvaluationService(storage_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            service.load_predictions("1999-01-01")
