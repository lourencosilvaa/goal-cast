"""Two analysis back-ends behind one interface.

The NVIDIA key has been storable since the Settings page grew a field for it,
and nothing has ever read it: ``/api/ai/analyze`` fetched the *Gemini* key
unconditionally and constructed a ``genai.Client``. A settings field that
encrypts a secret and then ignores it is worse than no field — it reports
success for something that cannot work.

So the choice of back-end becomes an interface with two implementations. The
tests below care about three things: that each provider reads *its own* key,
that a missing key is refused before any request goes out, and that upstream
failures arrive as something the API layer can turn into an honest status.
"""

import pytest

from config.config_loader import AiConfig, AiProviderConfig
from src.backend.services.ai_providers import (
    AnalysisProvider,
    AnalysisRequest,
    GeminiAnalysisProvider,
    MissingProviderKeyError,
    NvidiaAnalysisProvider,
    ProviderUnavailable,
    UnknownProviderError,
    build_analysis_provider,
)

_CONFIG = AiConfig(
    default_provider="gemini",
    gemini=AiProviderConfig(
        key_service="gemini",
        default_model="gemini-2.5-flash",
        base_url="",
        timeout=60,
    ),
    nvidia=AiProviderConfig(
        key_service="nvidia",
        default_model="meta/llama-3.3-70b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        timeout=60,
    ),
)


class _Keys:
    """Stands in for the encrypted key store."""

    def __init__(self, **keys: str):
        self._keys = keys
        self.asked: list[str] = []

    def get_user_key(self, user_id: str, service: str = "gemini") -> str | None:
        self.asked.append(service)
        return self._keys.get(service)


class _Response:
    def __init__(self, body, status_code: int = 200):
        self._body = body
        self.status_code = status_code
        self.text = str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Transport:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict, dict]] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, dict(json or {}), dict(headers or {})))
        response = self._responses.pop(0) if self._responses else _Response({})
        if isinstance(response, Exception):
            raise response
        return response


def _request(model: str = "") -> AnalysisRequest:
    return AnalysisRequest(prompt="Quem ganha?", model=model, user_id="u1")


def _nvidia_body(text: str = "O Porto ganha.") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class TestSelection:
    def test_the_named_provider_is_built(self):
        provider = build_analysis_provider("nvidia", _CONFIG, _Keys())
        assert isinstance(provider, NvidiaAnalysisProvider)

    def test_an_unnamed_provider_falls_back_to_the_configured_default(self):
        provider = build_analysis_provider("", _CONFIG, _Keys())
        assert isinstance(provider, GeminiAnalysisProvider)

    def test_the_default_is_configuration_not_a_constant(self):
        config = _CONFIG.model_copy(update={"default_provider": "nvidia"})
        assert isinstance(
            build_analysis_provider("", config, _Keys()), NvidiaAnalysisProvider
        )

    def test_an_unknown_provider_is_refused_by_name(self):
        with pytest.raises(UnknownProviderError, match="acme"):
            build_analysis_provider("acme", _CONFIG, _Keys())

    def test_both_providers_share_the_interface(self):
        for name in ("gemini", "nvidia"):
            assert isinstance(
                build_analysis_provider(name, _CONFIG, _Keys()), AnalysisProvider
            )


class TestNvidiaProvider:
    def _provider(self, *responses, keys=None) -> NvidiaAnalysisProvider:
        return NvidiaAnalysisProvider(
            _CONFIG.nvidia,
            keys if keys is not None else _Keys(nvidia="nv-key"),
            transport=_Transport(*responses),
        )

    def test_the_analysis_text_is_returned(self):
        provider = self._provider(_Response(_nvidia_body()))
        assert provider.analyse(_request()) == "O Porto ganha."

    def test_it_reads_the_nvidia_key_not_the_gemini_one(self):
        keys = _Keys(nvidia="nv-key", gemini="g-key")
        provider = self._provider(_Response(_nvidia_body()), keys=keys)
        provider.analyse(_request())
        assert keys.asked == ["nvidia"]

    def test_the_key_travels_as_a_bearer_token(self):
        provider = self._provider(_Response(_nvidia_body()))
        provider.analyse(_request())
        assert provider._transport.calls[0][2]["Authorization"] == "Bearer nv-key"

    def test_the_openai_compatible_path_is_used(self):
        provider = self._provider(_Response(_nvidia_body()))
        provider.analyse(_request())
        assert provider._transport.calls[0][0].endswith("/chat/completions")

    def test_the_prompt_is_sent_as_a_user_message(self):
        provider = self._provider(_Response(_nvidia_body()))
        provider.analyse(_request())
        body = provider._transport.calls[0][1]
        assert body["messages"][-1] == {"role": "user", "content": "Quem ganha?"}

    def test_the_configured_model_is_used_when_none_is_asked_for(self):
        provider = self._provider(_Response(_nvidia_body()))
        provider.analyse(_request())
        assert provider._transport.calls[0][1]["model"] == (
            "meta/llama-3.3-70b-instruct"
        )

    def test_an_explicit_model_wins(self):
        provider = self._provider(_Response(_nvidia_body()))
        provider.analyse(_request(model="deepseek-ai/deepseek-r1"))
        assert provider._transport.calls[0][1]["model"] == "deepseek-ai/deepseek-r1"

    def test_a_missing_key_is_refused_before_any_request(self):
        provider = self._provider(keys=_Keys())
        with pytest.raises(MissingProviderKeyError, match="NVIDIA"):
            provider.analyse(_request())
        assert provider._transport.calls == []

    def test_a_rejected_key_is_reported_as_unauthorised(self):
        provider = self._provider(_Response({}, status_code=401))
        with pytest.raises(ProviderUnavailable) as raised:
            provider.analyse(_request())
        assert raised.value.status_code == 401

    def test_a_rate_limit_keeps_its_status(self):
        """Retrying in a second is the right advice; retrying a 401 is not."""
        provider = self._provider(_Response({}, status_code=429))
        with pytest.raises(ProviderUnavailable) as raised:
            provider.analyse(_request())
        assert raised.value.status_code == 429

    def test_a_transport_failure_is_reported_as_unavailable(self):
        provider = self._provider(ConnectionError("no route"))
        with pytest.raises(ProviderUnavailable):
            provider.analyse(_request())

    def test_an_unreadable_body_is_reported(self):
        provider = self._provider(_Response(ValueError("not json")))
        with pytest.raises(ProviderUnavailable):
            provider.analyse(_request())

    def test_a_response_with_no_choices_is_reported(self):
        """An empty answer must not reach the UI as an empty analysis box."""
        provider = self._provider(_Response({"choices": []}))
        with pytest.raises(ProviderUnavailable):
            provider.analyse(_request())

    def test_a_choice_with_no_content_is_reported(self):
        provider = self._provider(_Response({"choices": [{"message": {}}]}))
        with pytest.raises(ProviderUnavailable):
            provider.analyse(_request())


class TestGeminiProvider:
    def test_it_reads_the_gemini_key_not_the_nvidia_one(self):
        """The bug this module replaces, from the other side."""
        keys = _Keys(gemini="g-key", nvidia="nv-key")

        def _factory(api_key: str):
            raise RuntimeError(f"would have used {api_key}")

        provider = GeminiAnalysisProvider(
            _CONFIG.gemini, keys, client_factory=_factory
        )
        with pytest.raises(RuntimeError, match="would have used g-key"):
            provider.analyse(_request())
        assert keys.asked == ["gemini"]

    def test_a_missing_key_is_refused_before_a_client_is_built(self):
        built: list[str] = []

        def _factory(api_key: str):
            built.append(api_key)
            raise AssertionError("must not be reached")

        provider = GeminiAnalysisProvider(
            _CONFIG.gemini, _Keys(), client_factory=_factory
        )
        with pytest.raises(MissingProviderKeyError, match="Gemini"):
            provider.analyse(_request())
        assert built == []

    def test_the_analysis_text_is_returned(self):
        class _Models:
            def generate_content(self, model, contents):
                assert model == "gemini-2.5-flash"
                return type("R", (), {"text": "O Porto ganha."})()

        provider = GeminiAnalysisProvider(
            _CONFIG.gemini,
            _Keys(gemini="g-key"),
            client_factory=lambda key: type("C", (), {"models": _Models()})(),
        )
        assert provider.analyse(_request()) == "O Porto ganha."
