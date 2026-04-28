"""Edge case tests for InferenceService HTTP client."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.backend.services.inference_service import InferenceService


def _make_config(space_url: str = "https://user-space.hf.space") -> MagicMock:
    config = MagicMock()
    config.inference.enabled = True
    config.inference.space_url = space_url
    return config


class TestInferenceServiceEdgeCases:

    @patch("src.backend.services.inference_service.requests.post")
    def test_trailing_slash_in_space_url_is_stripped(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"predictions": []},
        )
        config = _make_config("https://user-space.hf.space/")
        svc = InferenceService(config)
        svc.run(target_date="28/04/2024")

        call_url = mock_post.call_args[0][0]
        assert call_url == "https://user-space.hf.space/infer"

    @patch("src.backend.services.inference_service.requests.post")
    def test_space_connection_error_propagates(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("unreachable")

        config = _make_config()
        svc = InferenceService(config)

        with pytest.raises(requests.exceptions.ConnectionError):
            svc.run(target_date="28/04/2024")

    @patch("src.backend.services.inference_service.requests.post")
    def test_space_timeout_propagates(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")

        config = _make_config()
        svc = InferenceService(config)

        with pytest.raises(requests.exceptions.Timeout):
            svc.run(target_date="28/04/2024")

    @patch("src.backend.services.inference_service.requests.post")
    def test_space_500_raises_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )
        mock_post.return_value = mock_response

        config = _make_config()
        svc = InferenceService(config)

        with pytest.raises(requests.exceptions.HTTPError):
            svc.run(target_date="28/04/2024")

    @patch("src.backend.services.inference_service.requests.post")
    def test_missing_predictions_key_returns_empty_list(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {},
        )
        config = _make_config()
        svc = InferenceService(config)
        result = svc.run(target_date="28/04/2024")

        assert result == []

    @patch("src.backend.services.inference_service.requests.post")
    def test_request_uses_120s_timeout(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"predictions": []},
        )
        config = _make_config()
        svc = InferenceService(config)
        svc.run()

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs.get("timeout") == 120

    def test_disabled_with_space_url_still_raises(self):
        config = _make_config("https://user-space.hf.space")
        config.inference.enabled = False
        svc = InferenceService(config)

        with pytest.raises(RuntimeError, match="disabled"):
            svc.run()
