"""Match analysis, from whichever back-end the user has a key for.

Until this module existed there was one path: ``/api/ai/analyze`` fetched the
*Gemini* key — unconditionally, because ``get_user_key`` defaults to it — and
built a ``genai.Client``. The Settings page had grown a field for an NVIDIA
key months earlier, and it encrypted that key, stored it, reported success,
and nothing ever read it. A credential field that does nothing is worse than
an absent one: it tells the user they have configured something.

So the back-end becomes an interface with two implementations, chosen per
request and defaulting from configuration. Each reads **its own** key, named
in config rather than assumed, which is the part that had gone wrong.

Failure is deliberately typed rather than generic. Three outcomes reach the
API layer and they call for different answers: no key (the user must add one),
a refused or rate-limited upstream (nothing to fix here, retry or wait), and
an unusable response (the provider answered but said nothing).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Protocol

import requests

from config.config_loader import AiConfig, AiProviderConfig


class MissingProviderKeyError(RuntimeError):
    """The user has no key for the provider they asked for."""


class UnknownProviderError(ValueError):
    """A provider name nothing implements."""


class ProviderUnavailable(RuntimeError):
    """The provider could not produce an analysis.

    Carries the upstream status so the API layer can pass a 401 or a 429
    through unchanged — collapsing them into one code would tell a user to
    retry when the real problem is a bad key.
    """

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class AnalysisRequest:
    """One analysis: the prompt, the model asked for, and who is asking."""

    prompt: str
    #: Empty means "whatever the provider's configuration defaults to".
    model: str
    user_id: str


class KeyStore(Protocol):
    """The encrypted per-user key store, as this module needs it."""

    def get_user_key(self, user_id: str, service: str = ...) -> str | None: ...


class AnalysisProvider(ABC):
    """A back-end that turns a prompt into a written analysis."""

    #: Human-facing name, used in the "no key" message the user reads.
    label: ClassVar[str] = "provider"

    def __init__(self, config: AiProviderConfig, keys: KeyStore) -> None:
        self._config = config
        self._keys = keys

    @abstractmethod
    def analyse(self, request: AnalysisRequest) -> str:
        """The analysis text, or raise."""

    # ── shared plumbing ──────────────────────────────────────────────────

    def _api_key(self, user_id: str) -> str:
        """This provider's key, or refuse before anything is sent.

        Reads the service named in configuration, never a default: the whole
        bug this module replaces was one provider reading another's key.
        """
        key = self._keys.get_user_key(user_id=user_id, service=self._config.key_service)
        if not key:
            raise MissingProviderKeyError(
                f"Chave {self.label} não configurada. Adiciona-a nas Definições."
            )
        return key

    def _model(self, request: AnalysisRequest) -> str:
        return request.model.strip() or self._config.default_model


class GeminiAnalysisProvider(AnalysisProvider):
    """Google Gemini, through the official SDK."""

    label: ClassVar[str] = "Gemini"

    def __init__(
        self,
        config: AiProviderConfig,
        keys: KeyStore,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(config, keys)
        #: Injected so a test never constructs a real client. ``None`` means
        #: the real SDK, imported lazily — it is a heavy import and only this
        #: provider needs it.
        self._client_factory = client_factory

    def analyse(self, request: AnalysisRequest) -> str:
        api_key = self._api_key(request.user_id)
        client = self._build_client(api_key)
        try:
            response = client.models.generate_content(
                model=self._model(request), contents=request.prompt
            )
        except Exception as exc:
            raise ProviderUnavailable(
                str(exc), status_code=getattr(exc, "code", 502) or 502
            ) from exc
        text = getattr(response, "text", "") or ""
        if not text.strip():
            raise ProviderUnavailable("O Gemini devolveu uma resposta vazia.")
        return text

    def _build_client(self, api_key: str) -> Any:
        if self._client_factory is not None:
            return self._client_factory(api_key)
        from google import genai

        return genai.Client(api_key=api_key)


class NvidiaAnalysisProvider(AnalysisProvider):
    """NVIDIA NIM, through its OpenAI-compatible chat completions endpoint.

    Raw HTTP rather than an SDK: the call is one POST and the shape is
    published, so the wire format stays visible in this repository instead of
    changing when a third-party package is upgraded — the same reasoning as
    :class:`src.scrapers.european.http.HttpTransport`.
    """

    label: ClassVar[str] = "NVIDIA"

    PATH: ClassVar[str] = "/chat/completions"
    #: Statuses worth passing through unchanged rather than folding into 502.
    PASS_THROUGH: ClassVar[tuple[int, ...]] = (400, 401, 403, 404, 429)
    OK: ClassVar[int] = 200

    def __init__(
        self,
        config: AiProviderConfig,
        keys: KeyStore,
        transport: Any = None,
    ) -> None:
        super().__init__(config, keys)
        self._transport = transport or requests

    def analyse(self, request: AnalysisRequest) -> str:
        api_key = self._api_key(request.user_id)
        url = f"{self._config.base_url.rstrip('/')}{self.PATH}"
        try:
            response = self._transport.post(
                url,
                json={
                    "model": self._model(request),
                    "messages": [{"role": "user", "content": request.prompt}],
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
                timeout=self._config.timeout,
            )
        except Exception as exc:
            raise ProviderUnavailable(
                f"Não foi possível contactar a NVIDIA: {exc}"
            ) from exc

        return self._content(response)

    def _content(self, response: Any) -> str:
        status = getattr(response, "status_code", 0)
        if status != self.OK:
            raise ProviderUnavailable(
                self._error_detail(response, status),
                status_code=status if status in self.PASS_THROUGH else 502,
            )
        try:
            body = response.json()
        except Exception as exc:
            raise ProviderUnavailable(f"Resposta ilegível da NVIDIA: {exc}") from exc

        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ProviderUnavailable("A NVIDIA não devolveu qualquer análise.")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        text = str(content or "").strip()
        if not text:
            # An empty analysis box looks like a model with nothing to say; it
            # is actually a response we could not read.
            raise ProviderUnavailable("A NVIDIA devolveu uma análise vazia.")
        return text

    @staticmethod
    def _error_detail(response: Any, status: int) -> str:
        if status in (401, 403):
            return "Chave da NVIDIA inválida ou sem acesso a este modelo."
        if status == 404:
            return "Modelo NVIDIA não encontrado. Verifica o nome nas Definições."
        if status == 429:
            return "Limite de pedidos da NVIDIA excedido. Tenta daqui a pouco."
        return f"A NVIDIA respondeu HTTP {status}."


#: Provider name → the class implementing it. Adding a back-end is one entry
#: here plus a config block, and nothing in the API layer changes.
PROVIDER_TYPES: dict[str, type[AnalysisProvider]] = {
    "gemini": GeminiAnalysisProvider,
    "nvidia": NvidiaAnalysisProvider,
}


def build_analysis_provider(
    name: str, config: AiConfig, keys: KeyStore
) -> AnalysisProvider:
    """The named provider, or the configured default when none is named."""
    chosen = (name or config.default_provider).strip().lower()
    provider_type = PROVIDER_TYPES.get(chosen)
    if provider_type is None:
        raise UnknownProviderError(
            f"Fornecedor de análise desconhecido: {chosen!r}. "
            f"Disponíveis: {', '.join(sorted(PROVIDER_TYPES))}."
        )
    provider_config: AiProviderConfig = getattr(config, chosen)
    return provider_type(provider_config, keys)
