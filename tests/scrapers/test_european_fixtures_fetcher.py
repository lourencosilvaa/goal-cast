"""Tests for turning discovered fixtures into canonical, predictable matches.

The contract that matters: a fixture is predicted only when both teams are
*known*. Every provider spells clubs its own way — "FC Kairat", "Kairat
Almaty", "Sport Lisboa e Benfica" — and a fixture carries no country code to
scope the search with, so candidate teams have to be drawn from all 21 tracked
leagues. That widens suggestions, and suggestions are wrong often enough to
matter: searching every league offers "Sparta Rotterdam" for "AC Sparta Praha".

So acceptance stays exact-match-or-approved-alias, and anything else is queued
for a human rather than predicted for the wrong club.
"""

from datetime import date, datetime

from config.config_loader import EuropeanConfig, ProviderConfig, TeamAliasConfig
from src.scrapers.european.providers import EuropeanFixture, FixtureWindow
from src.scrapers.european_fixtures_fetcher import (
    EuropeanFixturesFetcher,
    build_provider_chain,
)
from src.teams.resolver import TeamAlias, TeamAliasRepository, TeamNameResolver

_REGISTRY = {
    "P1": ["Benfica", "Sp Lisbon", "Porto"],
    "E0": ["Arsenal", "Liverpool"],
    "N1": ["Sparta Rotterdam", "Ajax"],
}


class _Aliases(TeamAliasRepository):
    def __init__(self, aliases=None) -> None:
        self._aliases = aliases or []

    def get_aliases(self):
        return list(self._aliases)


class _Sink:
    def __init__(self) -> None:
        self.recorded = []

    def record(self, resolution):
        self.recorded.append(resolution)


class _Provider:
    name = "stub"

    def __init__(self, fixtures) -> None:
        self._fixtures = fixtures
        self.window: FixtureWindow | None = None
        self.competitions: list[str] | None = None

    def fetch(self, window, competitions):
        self.window = window
        self.competitions = list(competitions)
        return list(self._fixtures)


def _config(**overrides) -> EuropeanConfig:
    base = dict(
        competitions={"CL": "cl", "EL": "el"},
        country_leagues={"POR": ["P1"], "ENG": ["E0"], "NED": ["N1"]},
        alias_scope="EU",
        lookahead_days=7,
    )
    base.update(overrides)
    return EuropeanConfig(**base)


def _fixture(home: str, away: str, competition: str = "CL") -> EuropeanFixture:
    return EuropeanFixture(
        competition=competition,
        kickoff=datetime(2026, 8, 11, 15, 0),
        home_team=home,
        away_team=away,
        source="stub",
    )


def _fetcher(fixtures, aliases=None, sink=None) -> EuropeanFixturesFetcher:
    resolver = TeamNameResolver(_REGISTRY, _Aliases(aliases), TeamAliasConfig())
    return EuropeanFixturesFetcher(_config(), _Provider(fixtures), resolver, sink)


class TestResolution:
    def test_an_exact_name_resolves(self):
        result = _fetcher([_fixture("Arsenal", "Benfica")]).fetch()
        assert result[0].home_team == "Arsenal"
        assert result[0].away_team == "Benfica"

    def test_an_approved_alias_resolves(self):
        alias = TeamAlias("EU", "FC Kairat", "Benfica")
        result = _fetcher([_fixture("FC Kairat", "Arsenal")], [alias]).fetch()
        assert result[0].home_team == "Benfica"

    def test_an_unknown_name_does_not_resolve(self):
        result = _fetcher([_fixture("FC Kairat", "Arsenal")]).fetch()
        assert result[0].home_team is None
        assert not result[0].is_resolved

    def test_a_near_miss_is_never_auto_accepted(self):
        """'AC Sparta Praha' must not become 'Sparta Rotterdam'."""
        result = _fetcher([_fixture("AC Sparta Praha", "Arsenal")]).fetch()
        assert result[0].home_team is None

    def test_a_half_resolved_fixture_is_not_considered_resolved(self):
        result = _fetcher([_fixture("Arsenal", "FC Kairat")]).fetch()
        assert result[0].home_team == "Arsenal"
        assert not result[0].is_resolved

    def test_unresolved_names_are_listed(self):
        result = _fetcher([_fixture("FC Kairat", "PFC Levski Sofia")]).fetch()
        assert result[0].unresolved_names == ["FC Kairat", "PFC Levski Sofia"]

    def test_a_fully_resolved_fixture_lists_nothing_unresolved(self):
        result = _fetcher([_fixture("Arsenal", "Benfica")]).fetch()
        assert result[0].unresolved_names == []

    def test_the_original_fixture_is_preserved(self):
        result = _fetcher([_fixture("FC Kairat", "Arsenal")]).fetch()
        assert result[0].fixture.home_team == "FC Kairat"


class TestQueueing:
    def test_an_unknown_name_is_queued_for_review(self):
        sink = _Sink()
        _fetcher([_fixture("FC Kairat", "Arsenal")], sink=sink).fetch()
        assert [r.raw_name for r in sink.recorded] == ["FC Kairat"]

    def test_queued_names_use_the_shared_eu_scope(self):
        """No country is available from a fixture, so no country scope exists."""
        sink = _Sink()
        _fetcher([_fixture("FC Kairat", "Arsenal")], sink=sink).fetch()
        assert sink.recorded[0].scope == "EU"

    def test_resolved_names_are_not_queued(self):
        sink = _Sink()
        _fetcher([_fixture("Arsenal", "Benfica")], sink=sink).fetch()
        assert sink.recorded == []

    def test_a_missing_sink_is_tolerated(self):
        """Offline callers have no review queue and must still work."""
        assert _fetcher([_fixture("FC Kairat", "Arsenal")]).fetch()


class TestWindow:
    def test_the_window_starts_today_by_default(self):
        provider = _Provider([])
        resolver = TeamNameResolver(_REGISTRY, _Aliases(), TeamAliasConfig())
        EuropeanFixturesFetcher(_config(), provider, resolver).fetch()
        assert provider.window.start == date.today()

    def test_the_window_honours_the_configured_lookahead(self):
        provider = _Provider([])
        resolver = TeamNameResolver(_REGISTRY, _Aliases(), TeamAliasConfig())
        EuropeanFixturesFetcher(_config(), provider, resolver).fetch(
            start=date(2026, 8, 1)
        )
        assert provider.window.end == date(2026, 8, 8)

    def test_all_configured_competitions_are_requested_by_default(self):
        provider = _Provider([])
        resolver = TeamNameResolver(_REGISTRY, _Aliases(), TeamAliasConfig())
        EuropeanFixturesFetcher(_config(), provider, resolver).fetch()
        assert set(provider.competitions) == {"CL", "EL"}

    def test_an_explicit_competition_list_is_honoured(self):
        provider = _Provider([])
        resolver = TeamNameResolver(_REGISTRY, _Aliases(), TeamAliasConfig())
        EuropeanFixturesFetcher(_config(), provider, resolver).fetch(
            competitions=["CL"]
        )
        assert provider.competitions == ["CL"]


class TestBuildProviderChain:
    def _providers_config(self) -> EuropeanConfig:
        return _config(
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

    def test_providers_with_keys_are_built(self, monkeypatch):
        monkeypatch.setenv("TEST_ODDS_KEY", "a")
        monkeypatch.setenv("TEST_FD_KEY", "b")
        chain = build_provider_chain(self._providers_config(), transport=object())
        assert len(chain._providers) == 2

    def test_order_follows_configuration(self, monkeypatch):
        """Order is correctness here: first non-empty wins."""
        monkeypatch.setenv("TEST_ODDS_KEY", "a")
        monkeypatch.setenv("TEST_FD_KEY", "b")
        chain = build_provider_chain(self._providers_config(), transport=object())
        assert [p.name for p in chain._providers] == [
            "the-odds-api",
            "football-data.org",
        ]

    def test_a_provider_without_its_key_is_skipped(self, monkeypatch):
        monkeypatch.setenv("TEST_ODDS_KEY", "a")
        monkeypatch.delenv("TEST_FD_KEY", raising=False)
        chain = build_provider_chain(self._providers_config(), transport=object())
        assert [p.name for p in chain._providers] == ["the-odds-api"]

    def test_a_disabled_provider_is_skipped(self, monkeypatch):
        monkeypatch.setenv("TEST_ODDS_KEY", "a")
        monkeypatch.setenv("TEST_FD_KEY", "b")
        config = self._providers_config()
        config.providers["odds_api"].enabled = False
        chain = build_provider_chain(config, transport=object())
        assert [p.name for p in chain._providers] == ["football-data.org"]

    def test_an_unknown_provider_name_is_ignored(self, monkeypatch):
        config = self._providers_config()
        config.provider_order = ["not_a_real_provider"]
        assert build_provider_chain(config, transport=object())._providers == []

    def test_no_configuration_yields_an_empty_chain(self):
        assert build_provider_chain(_config(), transport=object())._providers == []
