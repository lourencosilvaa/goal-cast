"""Assembling the service from configuration and environment.

Everything decided here is decided *once*, at boot, from two injected inputs:
the validated config object and an environment mapping. Nothing below this
module reads ``os.environ`` or opens a YAML file, which is what lets a test
state the entire world a service runs in — and what makes "which providers are
active?" answerable by reading the config rather than the code.

Adding a source is a config entry plus one line in the registries below.
"""

import os
from dataclasses import dataclass
from typing import Mapping

from config.config_loader import (
    Config,
    ResultsConfig,
    ResultsProviderConfig,
)
from src.results_service.auth import service_api_key
from src.results_service.repository import JsonResultsRepository
from src.results_service.service import ResultsCatalogue, ResultsService
from src.scrapers.european.http import HttpTransport
from src.scrapers.results.chained import ChainedHistoryProvider, ChainedLiveProvider
from src.scrapers.results.flashscore_live import FlashscoreLiveProvider
from src.scrapers.results.football_data import (
    FootballDataHistoryProvider,
    FootballDataLiveProvider,
)
from src.scrapers.results.live_tracker import LiveResultsTracker
from src.scrapers.results.local_corpus import LocalCorpusHistoryProvider

#: Config key → the class implementing it, per chain. Two registries because
#: the same key means different classes in each: ``football_data`` is a
#: history provider in one and a live provider in the other.
HISTORY_PROVIDER_TYPES: dict[str, type] = {
    "football_data": FootballDataHistoryProvider,
}
LIVE_PROVIDER_TYPES: dict[str, type] = {
    "football_data": FootballDataLiveProvider,
}


class ResultsDisabledError(RuntimeError):
    """``results.enabled`` is false, so this service has nothing to serve.

    Starting anyway would answer every request with an empty board — a
    deployment mistake that looks exactly like a quiet weekend.
    """


@dataclass(frozen=True)
class FlashscoreClientSettings:
    """The four values the Playwright client reads, gathered in one object.

    They come from two different config sections (``scrapers`` and
    ``scrapers.flashscore``), and the client predates both — it takes a single
    object with these attributes.
    """

    base_url: str
    user_agent: str
    request_timeout: int
    leagues: dict[str, str]


@dataclass(frozen=True)
class ServiceRuntime:
    """Everything the app factory needs, assembled and validated."""

    service: ResultsService
    api_key: str


def build_runtime(
    config: Config, env: Mapping[str, str] | None = None
) -> ServiceRuntime:
    """The service plus its key, or an exception explaining why not.

    Both failure modes are deliberate boot failures rather than degraded
    running: a disabled track and a missing key are each a deployment that
    would otherwise look healthy while doing the wrong thing.
    """
    environment = os.environ if env is None else env
    if not config.results.enabled:
        raise ResultsDisabledError(
            "results.enabled is false. The results service would answer every "
            "request with an empty board; enable it or do not deploy this "
            "service."
        )
    return ServiceRuntime(
        service=build_service(config, environment),
        api_key=service_api_key(config.results.service, environment),
    )


def build_service(
    config: Config, env: Mapping[str, str] | None = None
) -> ResultsService:
    """Compose the chains, the tracker and the catalogue."""
    environment = os.environ if env is None else env
    results = config.results
    return ResultsService(
        history=ChainedHistoryProvider(_history_providers(results, environment)),
        tracker=LiveResultsTracker(
            ChainedLiveProvider(_live_providers(config, environment)),
            results.live,
            store=JsonResultsRepository(results.cache_dir),
        ),
        catalogue=_catalogue(config),
    )


# ── catalogue ────────────────────────────────────────────────────────────


def _catalogue(config: Config) -> ResultsCatalogue:
    """What may be asked for: the served leagues plus the UEFA competitions.

    ``data.leagues`` is deliberately *not* used. It is the training corpus and
    is wider than the product on purpose — offering a division the UI does not
    show would put matches on a board nobody can navigate to.
    """
    european = tuple(config.european.competitions) if config.european.enabled else ()
    leagues = tuple(dict.fromkeys((*config.data.served_leagues, *european)))
    return ResultsCatalogue(leagues=leagues, seasons=tuple(config.data.seasons))


# ── chains ───────────────────────────────────────────────────────────────


def _history_providers(results: ResultsConfig, env: Mapping[str, str]) -> list[object]:
    providers: list[object] = []
    for name in results.history_provider_order:
        if name == "local_corpus":
            if results.local_corpus.enabled:
                providers.append(
                    LocalCorpusHistoryProvider(
                        results.local_corpus,
                        HttpTransport(timeout=results.local_corpus.timeout),
                        cache_dir=results.cache_dir,
                    )
                )
            continue
        provider = _api_provider(HISTORY_PROVIDER_TYPES, name, results, env)
        if provider is not None:
            providers.append(provider)
    return providers


def _live_providers(config: Config, env: Mapping[str, str]) -> list[object]:
    results = config.results
    providers: list[object] = []
    for name in results.live_provider_order:
        if name == "flashscore":
            flashscore = _flashscore_provider(config)
            if flashscore is not None:
                providers.append(flashscore)
            continue
        provider = _api_provider(LIVE_PROVIDER_TYPES, name, results, env)
        if provider is not None:
            providers.append(provider)
    return providers


def _api_provider(
    registry: dict[str, type],
    name: str,
    results: ResultsConfig,
    env: Mapping[str, str],
) -> object | None:
    """One credential-bearing provider, or ``None`` when it cannot be built.

    A provider whose type is unregistered, whose config is missing, which is
    switched off, or whose key is unset is skipped rather than raised over. An
    absent key means "not set up" — the normal state in development — and the
    chain simply falls through to the next source.
    """
    provider_config: ResultsProviderConfig | None = results.providers.get(name)
    provider_type = registry.get(name)
    if provider_config is None or provider_type is None or not provider_config.enabled:
        return None
    api_key = str(env.get(provider_config.api_key_env, "") or "")
    if not api_key:
        print(f"[results] {name} skipped: {provider_config.api_key_env} is not set")
        return None
    transport = HttpTransport(
        timeout=provider_config.timeout, retries=provider_config.retries
    )
    provider: object = provider_type(provider_config, transport, api_key)
    return provider


def _flashscore_provider(config: Config) -> object | None:
    """The scraping fallback, built only when configuration asks for it.

    Needs no credential — it is a scraper, not an account — so the only gate
    is the kill switch, and the import is local so a deployment that leaves it
    off never loads Playwright at all.
    """
    provider_config = config.results.providers.get("flashscore")
    if provider_config is None or not provider_config.enabled:
        return None

    from src.scrapers.flashscore.playwright_client import FlashScorePlaywrightClient

    flashscore = config.scrapers.flashscore
    settings = FlashscoreClientSettings(
        base_url=flashscore.base_url,
        user_agent=config.scrapers.user_agent,
        request_timeout=config.scrapers.request_timeout,
        leagues=dict(flashscore.leagues),
    )
    return FlashscoreLiveProvider(provider_config, FlashScorePlaywrightClient(settings))


def load_runtime(env: Mapping[str, str] | None = None) -> ServiceRuntime:
    """Boot path: read the YAML, then assemble. Used by the app module only."""
    from config.config_loader import load_config

    return build_runtime(load_config(), env)


__all__ = [
    "FlashscoreClientSettings",
    "ResultsDisabledError",
    "ServiceRuntime",
    "build_runtime",
    "build_service",
    "load_runtime",
]
