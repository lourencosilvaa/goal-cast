"""A run that cannot discover any fixture must say so, not report success.

``european.required`` already means "fail loudly rather than degrade quietly"
for the corpus: a scheduled retrain that could not find it was overwriting a
calibrated model with an uncalibrated one and still exiting 0. An empty
provider chain is the same class of failure at the other end of the pipeline —
inference completes, uploads nothing, and reports success, because every
provider skipped itself over an unset key.

That is not hypothetical either. ``run-inference.yml`` had neither
``THE_ODDS_API_KEY`` nor ``FOOTBALL_DATA_API_KEY`` in its environment, so
every scheduled run would have discovered zero European fixtures and said
nothing about it.

Without ``required`` the silent skip stays correct: an unset key means "not
set up yet", which is a normal state during development.
"""

import pytest

from config.config_loader import EuropeanConfig, ProviderConfig
from src.scrapers.european_fixtures_fetcher import (
    NoFixtureProvidersError,
    build_provider_chain,
)


def _config(**overrides) -> EuropeanConfig:
    """Explicit settings — never the shipped defaults (§7.3)."""
    base = dict(
        competitions={"CL": "cl"},
        country_leagues={"ENG": ["E0"]},
        alias_scope="EU",
        lookahead_days=7,
        required=False,
        provider_order=["odds_api", "football_data"],
        providers={
            "odds_api": ProviderConfig(
                base_url="https://example.test/v4",
                api_key_env="TEST_ODDS_KEY",
                competitions={"CL": "soccer_cl"},
            ),
            "football_data": ProviderConfig(
                base_url="https://example.test/fd",
                api_key_env="TEST_FD_KEY",
                competitions={"CL": "CL"},
            ),
        },
    )
    base.update(overrides)
    return EuropeanConfig(**base)


def _no_keys(monkeypatch) -> None:
    monkeypatch.delenv("TEST_ODDS_KEY", raising=False)
    monkeypatch.delenv("TEST_FD_KEY", raising=False)


class TestRequiredMakesAnEmptyChainFatal:
    def test_no_keys_under_required_raises(self, monkeypatch):
        _no_keys(monkeypatch)
        with pytest.raises(NoFixtureProvidersError):
            build_provider_chain(_config(required=True), transport=object())

    def test_the_error_names_the_variables_that_are_missing(self, monkeypatch):
        """A failure that does not say what to set is a failure twice."""
        _no_keys(monkeypatch)
        with pytest.raises(NoFixtureProvidersError, match="TEST_ODDS_KEY"):
            build_provider_chain(_config(required=True), transport=object())

    def test_one_working_provider_is_enough(self, monkeypatch):
        _no_keys(monkeypatch)
        monkeypatch.setenv("TEST_ODDS_KEY", "a")
        chain = build_provider_chain(_config(required=True), transport=object())
        assert len(chain._providers) == 1

    def test_all_providers_disabled_under_required_also_raises(self, monkeypatch):
        """Keys present but nothing enabled discovers just as little."""
        monkeypatch.setenv("TEST_ODDS_KEY", "a")
        monkeypatch.setenv("TEST_FD_KEY", "b")
        config = _config(
            required=True,
            providers={
                "odds_api": ProviderConfig(
                    base_url="https://example.test/v4",
                    api_key_env="TEST_ODDS_KEY",
                    competitions={"CL": "soccer_cl"},
                    enabled=False,
                ),
            },
            provider_order=["odds_api"],
        )
        with pytest.raises(NoFixtureProvidersError):
            build_provider_chain(config, transport=object())

    def test_an_empty_provider_order_under_required_raises(self, monkeypatch):
        _no_keys(monkeypatch)
        config = _config(required=True, provider_order=[])
        with pytest.raises(NoFixtureProvidersError):
            build_provider_chain(config, transport=object())


class TestWithoutRequiredTheSkipStaysSilent:
    def test_no_keys_yields_an_empty_chain(self, monkeypatch):
        _no_keys(monkeypatch)
        chain = build_provider_chain(_config(required=False), transport=object())
        assert chain._providers == []

    def test_nothing_is_raised(self, monkeypatch):
        _no_keys(monkeypatch)
        build_provider_chain(_config(required=False), transport=object())

    def test_the_skip_is_still_reported_on_stdout(self, monkeypatch, capsys):
        """Silent to the exit code, not silent to the operator."""
        _no_keys(monkeypatch)
        build_provider_chain(_config(required=False), transport=object())
        assert "TEST_ODDS_KEY" in capsys.readouterr().out
