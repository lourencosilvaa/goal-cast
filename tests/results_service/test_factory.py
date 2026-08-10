"""Assembling the service from configuration and environment.

This is where "no hidden behaviour" is enforced in practice: which providers
exist, in which order, and with which credentials is decided here from the
config object and the environment map — both injected, neither read from a
global — so a test states the whole world it runs in.
"""

import pytest

from config.config_loader import (
    Config,
    LocalCorpusConfig,
    ResultsConfig,
    ResultsLiveConfig,
    ResultsProviderConfig,
    ResultsServiceConfig,
    load_config,
)
from src.results_service.factory import build_runtime, build_service
from src.scrapers.results.flashscore_live import FlashscoreLiveProvider
from src.scrapers.results.football_data import (
    FootballDataHistoryProvider,
    FootballDataLiveProvider,
)
from src.scrapers.results.local_corpus import LocalCorpusHistoryProvider


def _results(**overrides) -> ResultsConfig:
    defaults = dict(
        enabled=True,
        history_provider_order=["local_corpus", "football_data"],
        live_provider_order=["football_data", "flashscore"],
        live=ResultsLiveConfig(poll_interval_seconds=60, stale_after_seconds=300),
        cache_dir="datasets/cache/results",
        service=ResultsServiceConfig(api_key_env="RESULTS_SERVICE_API_KEY"),
        local_corpus=LocalCorpusConfig(
            enabled=True,
            base_url="https://example.test",
            search_dirs=["datasets/cache"],
            leagues=["P1"],
        ),
        providers={
            "football_data": ResultsProviderConfig(
                enabled=True,
                base_url="https://api.football-data.org/v4",
                api_key_env="FOOTBALL_DATA_API_KEY",
                competitions={"P1": "PPL"},
            ),
            "flashscore": ResultsProviderConfig(enabled=True),
        },
    )
    defaults.update(overrides)
    return ResultsConfig(**defaults)


def _config(config_path, **overrides) -> Config:
    base = load_config(config_path)
    return base.model_copy(update={"results": _results(**overrides)})


_ENV = {"FOOTBALL_DATA_API_KEY": "fd-key", "RESULTS_SERVICE_API_KEY": "svc-key"}


def _names(chain) -> list[str]:
    return [type(provider).__name__ for provider in chain.providers]


class TestChainAssembly:
    def test_history_providers_follow_the_configured_order(self, config_path):
        service = build_service(_config(config_path), env=_ENV)
        assert _names(service.history_chain) == [
            LocalCorpusHistoryProvider.__name__,
            FootballDataHistoryProvider.__name__,
        ]

    def test_reordering_the_configuration_reorders_the_chain(self, config_path):
        service = build_service(
            _config(config_path, history_provider_order=["football_data"]),
            env=_ENV,
        )
        assert _names(service.history_chain) == [FootballDataHistoryProvider.__name__]

    def test_live_providers_follow_the_configured_order(self, config_path):
        service = build_service(_config(config_path), env=_ENV)
        assert _names(service.live_chain) == [
            FootballDataLiveProvider.__name__,
            FlashscoreLiveProvider.__name__,
        ]

    def test_a_disabled_provider_is_left_out_of_the_chain(self, config_path):
        config = _config(
            config_path,
            providers={
                "football_data": ResultsProviderConfig(
                    enabled=True, api_key_env="FOOTBALL_DATA_API_KEY"
                ),
                "flashscore": ResultsProviderConfig(enabled=False),
            },
        )
        service = build_service(config, env=_ENV)
        assert FlashscoreLiveProvider.__name__ not in _names(service.live_chain)

    def test_a_provider_whose_key_is_unset_is_left_out(self, config_path):
        """An absent key means "not set up" — a normal development state, not
        a failure to raise in the middle of a request."""
        service = build_service(
            _config(config_path), env={"RESULTS_SERVICE_API_KEY": "svc-key"}
        )
        assert FootballDataLiveProvider.__name__ not in _names(service.live_chain)

    def test_the_keyless_fallback_survives_a_missing_api_key(self, config_path):
        service = build_service(
            _config(config_path), env={"RESULTS_SERVICE_API_KEY": "svc-key"}
        )
        assert FlashscoreLiveProvider.__name__ in _names(service.live_chain)

    def test_a_provider_name_with_no_registered_class_is_skipped(self, config_path):
        """Config can name a source the code does not implement yet; that is a
        chain one shorter, not a crash at boot."""
        config = _config(
            config_path,
            live_provider_order=["football_data", "flashscore"],
            providers={
                "football_data": ResultsProviderConfig(
                    enabled=True, api_key_env="FOOTBALL_DATA_API_KEY"
                ),
                "flashscore": ResultsProviderConfig(enabled=True),
                "sofascore": ResultsProviderConfig(enabled=True),
            },
        )
        service = build_service(
            config.model_copy(
                update={
                    "results": config.results.model_copy(
                        update={
                            "live_provider_order": [
                                "sofascore",
                                "football_data",
                            ]
                        }
                    )
                }
            ),
            env=_ENV,
        )
        assert _names(service.live_chain) == [FootballDataLiveProvider.__name__]

    def test_the_local_corpus_needs_no_credential_either(self, config_path):
        service = build_service(
            _config(config_path), env={"RESULTS_SERVICE_API_KEY": "svc-key"}
        )
        assert LocalCorpusHistoryProvider.__name__ in _names(service.history_chain)


class TestCatalogue:
    def test_the_served_leagues_are_what_may_be_asked_for(self, config_path):
        config = _config(config_path)
        service = build_service(config, env=_ENV)
        for code in config.data.served_leagues:
            assert code in service.catalogue.leagues

    def test_the_european_competitions_are_askable_too(self, config_path):
        config = _config(config_path)
        service = build_service(config, env=_ENV)
        for code in config.european.competitions:
            assert code in service.catalogue.leagues

    def test_a_league_that_is_trained_on_but_not_served_is_not_askable(
        self, config_path
    ):
        """``data.leagues`` is the training corpus and deliberately wider than
        the product; the results API must offer only what the product does."""
        config = _config(config_path)
        service = build_service(config, env=_ENV)
        hidden = set(config.data.leagues) - set(config.data.served_leagues)
        assert not (hidden & set(service.catalogue.leagues))

    def test_the_configured_seasons_are_what_may_be_asked_for(self, config_path):
        config = _config(config_path)
        service = build_service(config, env=_ENV)
        assert service.catalogue.seasons == tuple(config.data.seasons)


class TestRuntime:
    def test_the_runtime_carries_the_service_key(self, config_path):
        runtime = build_runtime(_config(config_path), env=_ENV)
        assert runtime.api_key == "svc-key"

    def test_a_missing_service_key_stops_the_boot(self, config_path):
        from src.results_service.auth import MissingServiceKeyError

        with pytest.raises(MissingServiceKeyError):
            build_runtime(_config(config_path), env={"FOOTBALL_DATA_API_KEY": "k"})

    def test_a_disabled_results_track_refuses_to_start_the_service(self, config_path):
        """Running the results service with ``results.enabled: false`` is a
        deployment mistake, and serving empty answers would hide it."""
        from src.results_service.factory import ResultsDisabledError

        with pytest.raises(ResultsDisabledError):
            build_runtime(_config(config_path, enabled=False), env=_ENV)
