"""Validation rules for the results system's configuration.

Two config objects live here because two processes read them: the dedicated
results service reads ``results:``, and the main app reads
``results_gateway:``. Neither may fall back silently — a provider order naming
a provider that does not exist would produce an empty chain, and an empty chain
is indistinguishable from "no matches today" at the API boundary (§7.4).
"""

import pytest
from pydantic import ValidationError

from config.config_loader import (
    Config,
    ResultsConfig,
    ResultsGatewayConfig,
    ResultsLiveConfig,
    ResultsProviderConfig,
    load_config,
)


def _providers() -> dict[str, ResultsProviderConfig]:
    return {
        "football_data": ResultsProviderConfig(
            enabled=True,
            base_url="https://api.football-data.org/v4",
            api_key_env="FOOTBALL_DATA_API_KEY",
            timeout=15,
            competitions={"P1": "PPL"},
        ),
        "flashscore": ResultsProviderConfig(enabled=False),
    }


def _results(**overrides) -> ResultsConfig:
    defaults = dict(
        enabled=True,
        history_provider_order=["local_corpus", "football_data"],
        live_provider_order=["football_data"],
        live=ResultsLiveConfig(poll_interval_seconds=60, stale_after_seconds=300),
        cache_dir="datasets/cache/results",
        providers=_providers(),
    )
    defaults.update(overrides)
    return ResultsConfig(**defaults)


class TestProviderOrders:
    def test_a_valid_order_is_accepted(self):
        assert _results().live_provider_order == ["football_data"]

    def test_live_order_naming_an_unconfigured_provider_is_refused(self):
        with pytest.raises(ValidationError, match="live_provider_order"):
            _results(live_provider_order=["football_data", "nowhere"])

    def test_history_order_naming_an_unconfigured_provider_is_refused(self):
        with pytest.raises(ValidationError, match="history_provider_order"):
            _results(history_provider_order=["ghost"])

    def test_local_corpus_is_a_valid_history_source_without_a_provider_entry(self):
        """It reads CSVs, so it has its own config block rather than a
        ``providers`` entry — the order must still be allowed to name it."""
        assert "local_corpus" in _results().history_provider_order

    def test_an_empty_live_order_is_refused(self):
        with pytest.raises(ValidationError):
            _results(live_provider_order=[])


class TestLiveWindows:
    def test_stale_after_must_not_precede_the_poll_interval(self):
        """A snapshot may not be declared stale before it is even refreshed."""
        with pytest.raises(ValidationError, match="stale_after_seconds"):
            ResultsLiveConfig(poll_interval_seconds=300, stale_after_seconds=60)

    def test_equal_windows_are_allowed(self):
        window = ResultsLiveConfig(poll_interval_seconds=60, stale_after_seconds=60)
        assert window.stale_after_seconds == 60

    def test_a_non_positive_poll_interval_is_refused(self):
        with pytest.raises(ValidationError):
            ResultsLiveConfig(poll_interval_seconds=0, stale_after_seconds=60)


class TestApiKeysStayOutOfTheFile:
    def test_a_provider_carries_only_the_env_var_name(self):
        provider = _providers()["football_data"]
        assert provider.api_key_env == "FOOTBALL_DATA_API_KEY"
        assert not hasattr(provider, "api_key")

    def test_the_gateway_carries_only_env_var_names(self):
        gateway = ResultsGatewayConfig(
            base_url_env="RESULTS_SERVICE_URL",
            api_key_env="RESULTS_SERVICE_API_KEY",
            timeout=10,
        )
        assert gateway.base_url_env == "RESULTS_SERVICE_URL"
        assert not hasattr(gateway, "base_url")


class TestShippedConfiguration:
    def test_config_yaml_loads_with_a_results_section(self, config_path):
        config: Config = load_config(config_path)
        assert config.results.enabled is True

    def test_every_mapped_live_league_is_a_real_league_code(self, config_path):
        """Guards typos. A code nobody serves would silently never be asked
        for; football-data.org covers only some of the served leagues (no
        Scotland, Belgium, Turkey or Greece), and the rest are honestly absent
        rather than mapped to an invented code."""
        config = load_config(config_path)
        mapped = config.results.providers["football_data"].competitions
        known = set(config.data.leagues) | set(config.european.competitions)
        assert [code for code in mapped if code not in known] == []

    def test_the_portuguese_league_is_mapped_for_live_results(self, config_path):
        config = load_config(config_path)
        assert config.results.providers["football_data"].competitions["P1"] == "PPL"

    def test_local_corpus_leagues_all_have_a_football_data_feed(self, config_path):
        """``local_corpus.leagues`` drives CSV downloads — a UEFA competition
        there would request a file that does not exist on every history call."""
        config = load_config(config_path)
        unknown = [
            code
            for code in config.results.local_corpus.leagues
            if code not in config.data.leagues
        ]
        assert unknown == []

    def test_config_yaml_declares_the_gateway_env_vars(self, config_path):
        config = load_config(config_path)
        assert config.results_gateway.base_url_env
        assert config.results_gateway.api_key_env
