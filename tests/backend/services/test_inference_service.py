"""Tests for InferenceService as an async httpx client calling the HF Space."""

from unittest.mock import MagicMock

import pytest

from src.backend.services.inference_service import InferenceService
from tests.backend.services.conftest import FakeAsyncClient, make_response


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

    @pytest.mark.asyncio
    async def test_run_posts_to_space_infer_endpoint(self, patch_async_client):
        client = patch_async_client(
            FakeAsyncClient(
                make_response(
                    {
                        "predictions": [
                            {
                                "home_team": "Arsenal",
                                "away_team": "Chelsea",
                                "predicted_outcome": "Home Win",
                                "confidence": 0.6,
                            }
                        ]
                    }
                )
            )
        )
        svc = InferenceService(_make_config("https://user-space.hf.space"))
        await svc.run(target_date="28/04/2024", league_codes=["E0"])

        assert client.last_url == "https://user-space.hf.space/infer"

    @pytest.mark.asyncio
    async def test_run_sends_date_and_league_codes(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response({"predictions": []})))
        svc = InferenceService(_make_config())
        await svc.run(target_date="28/04/2024", league_codes=["E0", "SP1"])

        assert client.last_json["date"] == "28/04/2024"
        assert client.last_json["league_codes"] == ["E0", "SP1"]

    @pytest.mark.asyncio
    async def test_run_returns_predictions_list(self, patch_async_client):
        predictions = [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "predicted_outcome": "Home Win",
                "confidence": 0.6,
            },
        ]
        patch_async_client(FakeAsyncClient(make_response({"predictions": predictions})))
        svc = InferenceService(_make_config())
        assert await svc.run(target_date="28/04/2024") == predictions

    @pytest.mark.asyncio
    async def test_run_returns_empty_list_when_no_predictions(
        self, patch_async_client
    ):
        patch_async_client(FakeAsyncClient(make_response({"predictions": []})))
        svc = InferenceService(_make_config())
        assert await svc.run(target_date="28/04/2024") == []

    @pytest.mark.asyncio
    async def test_run_raises_when_inference_disabled(self):
        config = _make_config()
        config.inference.enabled = False
        svc = InferenceService(config)

        with pytest.raises(RuntimeError, match="disabled"):
            await svc.run()

    @pytest.mark.asyncio
    async def test_run_sends_none_league_codes_when_not_specified(
        self, patch_async_client
    ):
        client = patch_async_client(FakeAsyncClient(make_response({"predictions": []})))
        svc = InferenceService(_make_config())
        await svc.run(target_date="28/04/2024")

        assert client.last_json["league_codes"] is None

    @pytest.mark.asyncio
    async def test_run_raises_when_space_url_is_empty(self):
        svc = InferenceService(_make_config(space_url=""))
        with pytest.raises(RuntimeError, match="HF_SPACE_URL"):
            await svc.run()
