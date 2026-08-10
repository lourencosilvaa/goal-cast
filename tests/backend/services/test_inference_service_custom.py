"""Tests for InferenceService.predict_custom() and get_teams()."""

from unittest.mock import MagicMock

import httpx
import pytest

from src.backend.services.inference_service import (
    InferenceService,
    PredictionRefused,
)
from tests.backend.services.conftest import FakeAsyncClient, make_response

_CUSTOM_RESULT = {
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "predicted_outcome": "Home Win",
    "confidence": 0.65,
    "probabilities": {"home_win": 0.65, "draw": 0.2, "away_win": 0.15},
    "league": "Premier League",
}


def _make_config(
    enabled: bool = True, space_url: str = "https://user.hf.space"
) -> MagicMock:
    config = MagicMock()
    config.inference.enabled = enabled
    config.inference.space_url = space_url
    return config


class TestInferenceServicePredictCustom:

    @pytest.mark.asyncio
    async def test_calls_hf_space_predict_custom(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response(_CUSTOM_RESULT)))
        svc = InferenceService(_make_config())
        result = await svc.predict_custom("Arsenal", "Chelsea", "E0")

        assert client.last_url == "https://user.hf.space/predict-custom"
        assert result["predicted_outcome"] == "Home Win"

    @pytest.mark.asyncio
    async def test_sends_correct_payload(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response(_CUSTOM_RESULT)))
        svc = InferenceService(_make_config())
        await svc.predict_custom("Arsenal", "Chelsea", "E0")

        assert client.last_json == {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "league_code": "E0",
            "away_league_code": None,
        }


class TestInferenceServiceCrossLeague:
    """The away side may sit in a different league from the home side.

    The Space decides which model that calls for; this service only has to
    carry the second code and not flatten the answer that comes back.
    """

    @pytest.mark.asyncio
    async def test_away_league_is_forwarded(self, patch_async_client):
        client = patch_async_client(FakeAsyncClient(make_response(_CUSTOM_RESULT)))
        svc = InferenceService(_make_config())
        await svc.predict_custom("Arsenal", "Benfica", "E0", away_league_code="P1")

        assert client.last_json["away_league_code"] == "P1"

    @pytest.mark.asyncio
    async def test_cross_league_fields_are_returned_verbatim(self, patch_async_client):
        patch_async_client(
            FakeAsyncClient(
                make_response(
                    {**_CUSTOM_RESULT, "model": "elo", "away_league": "Liga Portugal"}
                )
            )
        )
        svc = InferenceService(_make_config())
        result = await svc.predict_custom(
            "Arsenal", "Benfica", "E0", away_league_code="P1"
        )

        assert result["model"] == "elo"
        assert result["away_league"] == "Liga Portugal"

    @pytest.mark.asyncio
    async def test_refusal_becomes_prediction_refused(self, patch_async_client):
        """A stated reason is not an outage, and must not be reported as one.

        ``raise_for_status`` would give an ``HTTPStatusError`` whose message is
        the status line — the reason the Space took the trouble to write would
        be lost on the way to the user.
        """
        patch_async_client(
            FakeAsyncClient(
                make_response({"detail": "no history for 'Kairat'"}, status_code=422)
            )
        )
        svc = InferenceService(_make_config())
        with pytest.raises(PredictionRefused, match="Kairat"):
            await svc.predict_custom("Arsenal", "Kairat", "E0", away_league_code="P1")

    @pytest.mark.asyncio
    async def test_refusal_without_a_readable_body_still_refuses(
        self, patch_async_client
    ):
        response = make_response(status_code=422)
        response.json.side_effect = ValueError("not json")
        patch_async_client(FakeAsyncClient(response))
        svc = InferenceService(_make_config())
        with pytest.raises(PredictionRefused):
            await svc.predict_custom("Arsenal", "Kairat", "E0", away_league_code="P1")

    @pytest.mark.asyncio
    async def test_an_empty_reason_is_admitted_not_invented(self, patch_async_client):
        patch_async_client(
            FakeAsyncClient(make_response({"detail": ""}, status_code=422))
        )
        svc = InferenceService(_make_config())
        with pytest.raises(PredictionRefused) as caught:
            await svc.predict_custom("Arsenal", "Kairat", "E0", away_league_code="P1")
        assert str(caught.value) == InferenceService.UNSTATED_REFUSAL

    @pytest.mark.asyncio
    async def test_a_refusal_is_still_a_runtime_error(self):
        """Callers written before refusals existed catch ``RuntimeError``, and
        must keep catching this — degraded, but not broken."""
        assert issubclass(PredictionRefused, RuntimeError)

    @pytest.mark.asyncio
    async def test_an_empty_away_league_is_forwarded_unchanged(
        self, patch_async_client
    ):
        """Interpreting it is the Space's job — it holds the corpus and the
        routing rule. Coercing it here would put that decision in two places."""
        client = patch_async_client(FakeAsyncClient(make_response(_CUSTOM_RESULT)))
        svc = InferenceService(_make_config())
        await svc.predict_custom("Arsenal", "Chelsea", "E0", away_league_code="")

        assert client.last_json["away_league_code"] == ""

    @pytest.mark.asyncio
    async def test_other_statuses_are_not_refusals(self, patch_async_client):
        """503 is the Space being down, which is a different problem."""
        patch_async_client(
            FakeAsyncClient(
                make_response(
                    status_error=httpx.HTTPStatusError(
                        "503",
                        request=httpx.Request("POST", "https://user.hf.space"),
                        response=httpx.Response(503),
                    ),
                    status_code=503,
                )
            )
        )
        svc = InferenceService(_make_config())
        with pytest.raises(httpx.HTTPStatusError):
            await svc.predict_custom("Arsenal", "Benfica", "E0", away_league_code="P1")

    @pytest.mark.asyncio
    async def test_disabled_raises_runtime_error(self):
        svc = InferenceService(_make_config(enabled=False))
        with pytest.raises(RuntimeError, match="disabled"):
            await svc.predict_custom("Arsenal", "Chelsea", "E0")

    @pytest.mark.asyncio
    async def test_hf_space_error_propagates(self, patch_async_client):
        patch_async_client(
            FakeAsyncClient(
                make_response(
                    status_error=httpx.HTTPStatusError(
                        "500",
                        request=httpx.Request("POST", "https://user.hf.space"),
                        response=httpx.Response(500),
                    )
                )
            )
        )
        svc = InferenceService(_make_config())
        with pytest.raises(httpx.HTTPStatusError):
            await svc.predict_custom("Arsenal", "Chelsea", "E0")


class TestInferenceServiceGetTeams:
    """get_teams() backs the RemoteTeamRepository behind /api/predictions/teams."""

    @pytest.mark.asyncio
    async def test_gets_teams_endpoint(self, patch_async_client):
        client = patch_async_client(
            FakeAsyncClient(make_response({"E0": ["Arsenal", "Chelsea"]}))
        )
        svc = InferenceService(_make_config())
        result = await svc.get_teams()

        assert client.last_call[0] == "GET"
        assert client.last_url == "https://user.hf.space/teams"
        assert result == {"E0": ["Arsenal", "Chelsea"]}

    @pytest.mark.asyncio
    async def test_disabled_raises_runtime_error(self):
        svc = InferenceService(_make_config(enabled=False))
        with pytest.raises(RuntimeError, match="disabled"):
            await svc.get_teams()

    @pytest.mark.asyncio
    async def test_empty_space_response_is_returned_verbatim(self, patch_async_client):
        """Interpreting {} is the repository's job, not the service's."""
        patch_async_client(FakeAsyncClient(make_response({})))
        svc = InferenceService(_make_config())
        assert await svc.get_teams() == {}

    @pytest.mark.asyncio
    async def test_connection_error_propagates(self, patch_async_client):
        patch_async_client(
            FakeAsyncClient(request_error=httpx.ConnectError("unreachable"))
        )
        svc = InferenceService(_make_config())
        with pytest.raises(httpx.ConnectError):
            await svc.get_teams()
