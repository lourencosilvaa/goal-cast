"""The main app's client for the dedicated results service.

An interface plus one HTTP implementation. The interface is what keeps the
microservice decision reversible: an ``InProcessResultsGateway`` that called
the provider chain directly would be a second implementation, and the router
above would not change a line. It is also why the backend can stay ignorant of
providers, Playwright and quotas — it knows two method names and a wire
contract.

**Every failure is stated.** A results service that is asleep, slow,
misconfigured or answering a shape we do not recognise raises, and the router
turns that into a 503 with a message. None of them may become an empty list:
"no matches" and "we could not find out" render as the same blank screen, and
only one of them is true.

Its own thin ``requests`` call rather than
:class:`src.scrapers.european.http.HttpTransport`, deliberately: importing
that would put ``src/scrapers`` — and eventually its dependencies — into the
application image, which is exactly what splitting the service out avoided.
Retries are absent for the same reason they are pointless here: this is one
hop inside the platform's own network, and a caller waiting on a page is
better served by a fast, honest failure than by a third attempt.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Sequence, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from config.config_loader import ResultsGatewayConfig
from src.contracts.results import HistoryResponse, LiveResultsResponse

#: Any of the shared response models, so ``_parse`` returns the model it was
#: handed rather than ``Any`` — the point of validating in the first place.
_Contract = TypeVar("_Contract", bound=BaseModel)


class ResultsServiceUnavailable(RuntimeError):
    """The results service could not answer. Surfaced as 503, never as empty."""


class ResultsRequestRejected(ValueError):
    """The service refused the request itself — an unknown league or season.

    Kept apart from unavailability because retrying will not help and the
    caller, not the deployment, is what needs to change.
    """


class MissingGatewayConfigError(RuntimeError):
    """A deployment variable this gateway needs is unset.

    An outage from where a user stands, but the message names the variable so
    whoever reads the logs is not left guessing.
    """


class ResultsGateway(ABC):
    """How the app asks for results, without knowing where they come from."""

    @abstractmethod
    def live(self, leagues: Sequence[str]) -> LiveResultsResponse:
        """Today's matches for ``leagues``; empty means every offered league."""

    @abstractmethod
    def history(self, league: str, season: str) -> HistoryResponse:
        """Finished matches for one league-season."""


class RequestsTransport:
    """One GET with a timeout. No retry policy, by design (see module docs)."""

    def __init__(self, timeout: int) -> None:
        self._timeout = timeout

    def get(
        self, url: str, params: dict | None = None, headers: dict | None = None
    ) -> Any:
        return requests.get(
            url, params=params or {}, headers=headers or {}, timeout=self._timeout
        )


class HttpResultsGateway(ResultsGateway):
    """Calls the results service over HTTP with the shared service key."""

    LIVE_PATH: ClassVar[str] = "/live"
    HISTORY_PATH: ClassVar[str] = "/history"
    AUTH_HEADER: ClassVar[str] = "X-API-Key"
    OK: ClassVar[int] = 200
    #: The service's "you asked for something I do not offer".
    REJECTED: ClassVar[int] = 422

    def __init__(
        self,
        config: ResultsGatewayConfig,
        transport: Any = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._config = config
        self._transport = transport or RequestsTransport(config.timeout)
        #: Injected so a test states the environment rather than inheriting it.
        self._env = os.environ if env is None else env

    def live(self, leagues: Sequence[str]) -> LiveResultsResponse:
        params = {"leagues": ",".join(leagues)} if leagues else {}
        body = self._get(self.LIVE_PATH, params)
        return self._parse(LiveResultsResponse, body)

    def history(self, league: str, season: str) -> HistoryResponse:
        body = self._get(self.HISTORY_PATH, {"league": league, "season": season})
        return self._parse(HistoryResponse, body)

    # ── the call ─────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict) -> Any:
        # Resolved before the try: a missing variable is a configuration
        # error of ours, and wrapping it as "unreachable" would send whoever
        # is debugging to look at the network.
        url = f"{self._base_url()}{path}"
        headers = {self.AUTH_HEADER: self._api_key()}
        try:
            response = self._transport.get(url, params=params, headers=headers)
        except Exception as exc:
            raise ResultsServiceUnavailable(
                f"results service unreachable at {url}: {exc}"
            ) from exc

        status_code = getattr(response, "status_code", 0)
        if status_code == self.REJECTED:
            raise ResultsRequestRejected(self._detail(response))
        if status_code != self.OK:
            # 401 lands here too: a rejected service key is our configuration's
            # fault, and reporting it as a transient outage would hide it.
            raise ResultsServiceUnavailable(
                f"results service returned HTTP {status_code} for {path}"
            )
        try:
            return response.json()
        except Exception as exc:
            raise ResultsServiceUnavailable(
                f"results service sent an unreadable body for {path}: {exc}"
            ) from exc

    @staticmethod
    def _parse(model: type[_Contract], body: Any) -> _Contract:
        """Validate against the shared contract.

        A body that does not fit is drift between two separately deployed
        services, and it has to surface here. Accepting it partially would put
        a half-populated screen in front of a user with nothing in the logs.
        """
        try:
            return model.model_validate(body)
        except ValidationError as exc:
            raise ResultsServiceUnavailable(
                f"results service sent an unexpected payload: {exc}"
            ) from exc

    @staticmethod
    def _detail(response: Any) -> str:
        try:
            body = response.json()
        except Exception:
            return "results service rejected the request"
        detail = body.get("detail") if isinstance(body, dict) else None
        return str(detail or "results service rejected the request")

    # ── environment ──────────────────────────────────────────────────────

    def _base_url(self) -> str:
        return self._required(self._config.base_url_env).rstrip("/")

    def _api_key(self) -> str:
        return self._required(self._config.api_key_env)

    def _required(self, name: str) -> str:
        value = str(self._env.get(name, "") or "").strip()
        if not value:
            raise MissingGatewayConfigError(
                f"{name} is not set, so the results service cannot be reached."
            )
        return value
