"""Discover upcoming UEFA club fixtures and resolve them to canonical teams.

Two responsibilities, kept apart on purpose:

* **Discovery** is delegated to a :class:`ChainedFixtureProvider`, so which
  API answers is a configuration question, not a code one.
* **Resolution** applies the same refuse-to-guess contract used everywhere
  else. A fixture name resolves only if it exactly matches a canonical team or
  an already-approved alias. Anything else is queued for human review and the
  fixture is reported as unresolved rather than predicted for the wrong club.

Why fixtures cannot use the corpus's country scoping: an openfootball result
carries the club's country, which is what makes ``EU-POR`` scoping safe. A
fixture from any of these APIs carries no country at all, so candidates would
have to be drawn from every tracked league — where "AC Sparta Praha" finds
"Sparta Rotterdam" and would silently attribute a Czech club's history to a
Dutch one. Rather than widen the guess, fixture names are queued under the
plain ``EU`` scope and suggestions stay advisory.
"""

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, ClassVar

from config.config_loader import EuropeanConfig, ProviderConfig
from src.scrapers.european.chained import ChainedFixtureProvider
from src.scrapers.european.football_data import FootballDataProvider
from src.scrapers.european.http import HttpTransport
from src.scrapers.european.odds_api import OddsApiProvider
from src.scrapers.european.providers import EuropeanFixture, FixtureWindow
from src.teams.resolver import TeamNameQuery, TeamNameResolver

#: Provider key in config → the class that implements it. Adding a source is a
#: config entry plus one line here, and nothing else changes.
PROVIDER_TYPES: dict[str, type] = {
    "odds_api": OddsApiProvider,
    "football_data": FootballDataProvider,
}


@dataclass(frozen=True)
class ResolvedFixture:
    """A fixture with canonical team names where they could be established."""

    fixture: EuropeanFixture
    home_team: str | None
    away_team: str | None

    @property
    def is_resolved(self) -> bool:
        return self.home_team is not None and self.away_team is not None

    @property
    def unresolved_names(self) -> list[str]:
        names = []
        if self.home_team is None:
            names.append(self.fixture.home_team)
        if self.away_team is None:
            names.append(self.fixture.away_team)
        return names


class NoFixtureProvidersError(RuntimeError):
    """Calibration was required but no provider could be built.

    Raised only when ``european.required`` is set, matching
    :class:`~src.corpus.european_corpus.MissingEuropeanCorpusError`: a chain
    with no providers discovers no fixtures, so the run completes, uploads
    nothing and reports success. That silent success is exactly what
    ``required`` exists to remove — and it is what a scheduled inference run
    would have done, since the workflow set neither provider's API key.
    """


def build_provider_chain(
    config: EuropeanConfig, transport: Any = None
) -> ChainedFixtureProvider:
    """Assemble the configured providers, in configured order.

    A provider whose key is unknown, whose type is unregistered, or whose
    environment variable is unset is skipped — an absent key means "not set
    up", which is a normal state during development, not an error worth
    raising in the middle of a fixture fetch.

    Unless ``european.required`` is set and *every* provider was skipped, in
    which case there is nothing left to fetch with and the run should say so
    rather than discover zero fixtures quietly.
    """
    providers = []
    skipped: list[str] = []
    for name in config.provider_order:
        provider_config: ProviderConfig | None = config.providers.get(name)
        provider_type = PROVIDER_TYPES.get(name)
        if provider_config is None or provider_type is None:
            continue
        if not provider_config.enabled:
            continue
        api_key = os.environ.get(provider_config.api_key_env, "")
        if not api_key:
            print(
                f"[fixtures] {name} skipped: {provider_config.api_key_env} is not set"
            )
            skipped.append(provider_config.api_key_env)
            continue
        client = transport or HttpTransport(
            timeout=provider_config.timeout, retries=provider_config.retries
        )
        providers.append(provider_type(provider_config, client, api_key))

    if not providers and config.required:
        raise NoFixtureProvidersError(
            "no fixture provider could be built, so no European fixture can be "
            "discovered. "
            + (
                f"Set one of: {', '.join(skipped)}."
                if skipped
                else "Every configured provider is disabled or unregistered."
            )
            + " In CI these come from repository secrets; set european.required "
            "to false to allow running without them."
        )
    return ChainedFixtureProvider(providers)


class EuropeanFixturesFetcher:
    """Lists upcoming European fixtures with canonical team names.

    The provider chain and the name resolver are both injected, so this is
    testable without network access or a registry file, and either can be
    replaced without touching the other.
    """

    #: Scope unresolved fixture names are queued under. Deliberately not
    #: country-qualified: see the module docstring.
    ALIAS_SCOPE: ClassVar[str] = "EU"

    def __init__(
        self,
        config: EuropeanConfig,
        provider: Any,
        resolver: TeamNameResolver,
        sink: Any = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._resolver = resolver
        self._sink = sink

    def window(self, start: date | None = None) -> FixtureWindow:
        first = start or date.today()
        return FixtureWindow(
            start=first, end=first + timedelta(days=self._config.lookahead_days)
        )

    def fetch(
        self, competitions: list[str] | None = None, start: date | None = None
    ) -> list[ResolvedFixture]:
        wanted = competitions or list(self._config.competitions)
        fixtures = self._provider.fetch(self.window(start), wanted)
        return [self._resolve(fixture) for fixture in fixtures]

    def _resolve(self, fixture: EuropeanFixture) -> ResolvedFixture:
        return ResolvedFixture(
            fixture=fixture,
            home_team=self._canonical(fixture, fixture.home_team),
            away_team=self._canonical(fixture, fixture.away_team),
        )

    def _canonical(self, fixture: EuropeanFixture, raw_name: str) -> str | None:
        """The canonical team, or None when a human must decide.

        Candidates span every tracked league because a fixture carries no
        country. That widens *suggestions*, never acceptance: the resolver
        returns a canonical name only on an exact match or an approved alias.
        """
        query = TeamNameQuery(
            league_code=fixture.competition,
            raw_name=raw_name,
            candidate_league_codes=self._all_leagues(),
            alias_scope=self.ALIAS_SCOPE,
            # Read from every country scope too. The corpus approved these
            # clubs under EU-BEL, EU-GRE and so on; a fixture has no country
            # to reproduce that with, and without this the same 99 clubs
            # would have to be approved a second time.
            alias_search_scopes=self._alias_search_scopes(),
        )
        resolution = self._resolver.resolve(query)
        if resolution.canonical:
            return resolution.canonical
        if self._sink is not None:
            self._sink.record(resolution)
        return None

    def _all_leagues(self) -> tuple[str, ...]:
        leagues: list[str] = []
        for codes in self._config.country_leagues.values():
            leagues.extend(codes)
        return tuple(dict.fromkeys(leagues))

    def _alias_search_scopes(self) -> tuple[str, ...]:
        """``EU`` plus every ``EU-<COUNTRY>`` scope the corpus approves into."""
        prefix = self._config.alias_scope
        return (
            prefix,
            *(f"{prefix}-{country}" for country in self._config.country_leagues),
        )
