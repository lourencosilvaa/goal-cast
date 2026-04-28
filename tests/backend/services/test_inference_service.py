"""Tests for InferenceService as HTTP client calling the HF Space."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.backend.services.inference_service import InferenceService


def _make_config(space_url: str = "https://user-space.hf.space") -> MagicMock:
    config = MagicMock()
    config.inference.enabled = True
    config.inference.space_url = space_url
    return config


class TestInferenceService:

    def test_init_stores_config(self):
        config = _make_config()
        svc = InferenceService(config)
        assert svc.config is config

    @patch("src.backend.services.inference_service.requests.post")
    def test_run_posts_to_space_infer_endpoint(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"predictions": [{"home_team": "Arsenal", "away_team": "Chelsea",
                                           "predicted_outcome": "Home Win", "confidence": 0.6}]},
        )
        config = _make_config("https://user-space.hf.space")
        svc = InferenceService(config)
        svc.run(target_date="28/04/2024", league_codes=["E0"])

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert call_url == "https://user-space.hf.space/infer"

    @patch("src.backend.services.inference_service.requests.post")
    def test_run_sends_date_and_league_codes(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"predictions": []},
        )
        config = _make_config()
        svc = InferenceService(config)
        svc.run(target_date="28/04/2024", league_codes=["E0", "SP1"])

        payload = mock_post.call_args[1]["json"]
        assert payload["date"] == "28/04/2024"
        assert payload["league_codes"] == ["E0", "SP1"]

    @patch("src.backend.services.inference_service.requests.post")
    def test_run_returns_predictions_list(self, mock_post):
        predictions = [
            {"home_team": "Arsenal", "away_team": "Chelsea",
             "predicted_outcome": "Home Win", "confidence": 0.6},
        ]
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"predictions": predictions},
        )
        config = _make_config()
        svc = InferenceService(config)
        result = svc.run(target_date="28/04/2024")

        assert result == predictions

    @patch("src.backend.services.inference_service.requests.post")
    def test_run_returns_empty_list_when_no_predictions(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"predictions": []},
        )
        config = _make_config()
        svc = InferenceService(config)
        result = svc.run(target_date="28/04/2024")

        assert result == []

    def test_run_raises_when_inference_disabled(self):
        config = _make_config()
        config.inference.enabled = False
        svc = InferenceService(config)

        with pytest.raises(RuntimeError, match="disabled"):
            svc.run()

    @patch("src.backend.services.inference_service.requests.post")
    def test_run_sends_none_league_codes_when_not_specified(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"predictions": []},
        )
        config = _make_config()
        svc = InferenceService(config)
        svc.run(target_date="28/04/2024")

        payload = mock_post.call_args[1]["json"]
        assert payload["league_codes"] is None
