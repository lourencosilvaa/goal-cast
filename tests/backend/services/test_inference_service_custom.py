"""Tests for InferenceService.predict_custom()."""

from unittest.mock import MagicMock, patch

import pytest

from src.backend.services.inference_service import InferenceService


def _make_config(enabled: bool = True, space_url: str = "https://user.hf.space") -> MagicMock:
    config = MagicMock()
    config.inference.enabled = enabled
    config.inference.space_url = space_url
    return config


class TestInferenceServicePredictCustom:

    @patch("src.backend.services.inference_service.requests.post")
    def test_calls_hf_space_predict_custom(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "home_team": "Arsenal", "away_team": "Chelsea",
                "predicted_outcome": "Home Win", "confidence": 0.65,
                "probabilities": {"home_win": 0.65, "draw": 0.2, "away_win": 0.15},
                "league": "Premier League",
            },
        )
        mock_post.return_value.raise_for_status = MagicMock()

        svc = InferenceService(_make_config())
        result = svc.predict_custom("Arsenal", "Chelsea", "E0")

        call_url = mock_post.call_args[0][0]
        assert "/predict-custom" in call_url
        assert result["predicted_outcome"] == "Home Win"

    @patch("src.backend.services.inference_service.requests.post")
    def test_sends_correct_payload(self, mock_post):
        mock_post.return_value = MagicMock(
            json=lambda: {"home_team": "A", "away_team": "B", "predicted_outcome": "Draw",
                          "confidence": 0.4, "probabilities": {}, "league": "E0"},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        svc = InferenceService(_make_config())
        svc.predict_custom("Arsenal", "Chelsea", "E0")

        payload = mock_post.call_args[1]["json"]
        assert payload["home_team"] == "Arsenal"
        assert payload["away_team"] == "Chelsea"
        assert payload["league_code"] == "E0"

    def test_disabled_raises_runtime_error(self):
        svc = InferenceService(_make_config(enabled=False))
        with pytest.raises(RuntimeError, match="disabled"):
            svc.predict_custom("Arsenal", "Chelsea", "E0")

    @patch("src.backend.services.inference_service.requests.post")
    def test_hf_space_error_propagates(self, mock_post):
        import requests as req
        mock_post.return_value = MagicMock()
        mock_post.return_value.raise_for_status.side_effect = req.exceptions.HTTPError("500")

        svc = InferenceService(_make_config())
        with pytest.raises(req.exceptions.HTTPError):
            svc.predict_custom("Arsenal", "Chelsea", "E0")
