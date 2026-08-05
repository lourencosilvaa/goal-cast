"""Canonical team-name resolution for scraped fixtures.

Every model, data set and statistic in this project is keyed by the
football-data.co.uk spelling of a team ("Sp Lisbon", "Man City"). The primary
fixture source is that same site, so names line up by construction — but the
FlashScore fallback spells them its own way ("Sporting CP", "Manchester City").

An unrecognised name used to pass straight through, and the pipeline then
found no matching history and quietly predicted from *league averages* — a
confident-looking prediction about a team it had never identified. This module
closes that hole by resolving names against the registry and refusing to guess:

* an **exact** match (case- and whitespace-insensitive) resolves;
* an **alias** resolves, but only one a human approved — either committed to
  the YAML seed (validated by code review) or approved by an admin in the UI
  (validated in Supabase);
* anything else comes back **unresolved**, carrying advisory suggestions for a
  human to accept or reject. Suggestions are never applied automatically:
  "Sporting", "Sp Lisbon" and "Sporting Gijon" are all close, and picking one
  by similarity is how the wrong team ends up in a prediction.

Callers are expected to skip unresolved fixtures rather than fall back — a
missing prediction is honest, a mis-attributed one is not.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import ClassVar

import yaml

from config.config_loader import TeamAliasConfig


@dataclass(frozen=True)
class TeamAlias:
    """An approved mapping from a scraped spelling to the canonical one."""

    league_code: str
    raw_name: str
    canonical_name: str


@dataclass(frozen=True)
class TeamNameQuery:
    """A scraped name to resolve within one competition."""

    league_code: str
    raw_name: str


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving one scraped name.

    ``canonical`` is set only when the name resolved; ``suggestions`` are
    advisory candidates for an admin to review, never applied automatically.
    """

    STATUS_EXACT: ClassVar[str] = "exact"
    STATUS_ALIAS: ClassVar[str] = "alias"
    STATUS_UNRESOLVED: ClassVar[str] = "unresolved"

    raw_name: str
    league_code: str
    canonical: str | None = None
    status: str = STATUS_UNRESOLVED
    suggestions: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.canonical is not None


@dataclass(frozen=True)
class FixtureResolution:
    """Outcome of resolving both sides of a fixture.

    ``resolved`` is true only when *both* names resolved — a fixture is only
    usable when the identity of both teams is certain.
    """

    home_team: str | None
    away_team: str | None
    unresolved: list[Resolution] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.home_team is not None and self.away_team is not None


class TeamAliasRepository(ABC):
    """Read-only source of approved team-name aliases."""

    @abstractmethod
    def get_aliases(self) -> list[TeamAlias]:
        """Return every approved alias, or ``[]`` when unavailable."""


class StaticTeamAliasRepository(TeamAliasRepository):
    """Aliases committed to a YAML file and shipped with the image."""

    ENCODING: ClassVar[str] = "utf-8"
    ALIASES_KEY: ClassVar[str] = "aliases"

    def __init__(self, seed_path: str | Path) -> None:
        self._seed_path = Path(seed_path)

    def get_aliases(self) -> list[TeamAlias]:
        try:
            raw = yaml.safe_load(self._seed_path.read_text(encoding=self.ENCODING))
        except (OSError, yaml.YAMLError):
            return []
        if not isinstance(raw, dict):
            return []
        by_league = raw.get(self.ALIASES_KEY)
        if not isinstance(by_league, dict):
            return []
        return [
            TeamAlias(
                league_code=str(league_code),
                raw_name=str(scraped),
                canonical_name=str(canonical),
            )
            for league_code, entries in by_league.items()
            if isinstance(entries, dict)
            for scraped, canonical in entries.items()
            if str(scraped).strip() and str(canonical).strip()
        ]


class ChainedTeamAliasRepository(TeamAliasRepository):
    """Every alias from every source, in order.

    Unlike the team *name* sources, aliases accumulate rather than override:
    the reviewed seed and the admin-approved table are both authoritative, and
    a source that fails simply contributes nothing.
    """

    def __init__(self, sources: Sequence[TeamAliasRepository]) -> None:
        self._sources = list(sources)

    def get_aliases(self) -> list[TeamAlias]:
        aliases: list[TeamAlias] = []
        for source in self._sources:
            try:
                aliases.extend(source.get_aliases())
            except Exception:
                continue
        return aliases


class UnresolvedNameSink(ABC):
    """Destination for names that need a human decision."""

    @abstractmethod
    def record(self, resolution: Resolution) -> None:
        """Queue an unresolved name for admin review."""


class NullUnresolvedNameSink(UnresolvedNameSink):
    """Discards unresolved names — used when no review queue is configured."""

    def record(self, resolution: Resolution) -> None:
        return None


class ListUnresolvedNameSink(UnresolvedNameSink):
    """Collects unresolved names in memory (tests, dry runs, reporting)."""

    def __init__(self) -> None:
        self.recorded: list[Resolution] = []

    def record(self, resolution: Resolution) -> None:
        self.recorded.append(resolution)


class TeamNameResolver:
    """Maps scraped team names onto canonical ones, or reports failure.

    ``canonical_teams`` is the registry (``{league_code: [team, ...]}``) and is
    injected, so the same resolver serves the offline pipeline (registry file)
    and the backend (live team repository) without knowing where it came from.
    """

    def __init__(
        self,
        canonical_teams: Mapping[str, Sequence[str]],
        alias_repository: TeamAliasRepository,
        config: TeamAliasConfig,
    ) -> None:
        self.config = config
        self._canonical = {code: list(teams) for code, teams in canonical_teams.items()}
        self._alias_repository = alias_repository
        self._alias_cache: dict[str, dict[str, str]] | None = None

    def resolve(self, query: TeamNameQuery) -> Resolution:
        """Resolve one scraped name within its competition."""
        canonical_names = self._canonical.get(query.league_code, [])
        key = self._normalise(query.raw_name)
        if not key or not canonical_names:
            return Resolution(
                raw_name=query.raw_name,
                league_code=query.league_code,
                suggestions=self._suggest(query.raw_name, canonical_names),
            )

        for name in canonical_names:
            if self._normalise(name) == key:
                return Resolution(
                    raw_name=query.raw_name,
                    league_code=query.league_code,
                    canonical=name,
                    status=Resolution.STATUS_EXACT,
                )

        alias_target = self._aliases().get(query.league_code, {}).get(key)
        # A stale alias must never inject a team the data set does not have.
        if alias_target is not None and alias_target in canonical_names:
            return Resolution(
                raw_name=query.raw_name,
                league_code=query.league_code,
                canonical=alias_target,
                status=Resolution.STATUS_ALIAS,
            )

        return Resolution(
            raw_name=query.raw_name,
            league_code=query.league_code,
            suggestions=self._suggest(query.raw_name, canonical_names),
        )

    def _aliases(self) -> dict[str, dict[str, str]]:
        """Approved aliases as ``{league: {normalised raw: canonical}}``."""
        if self._alias_cache is None:
            cache: dict[str, dict[str, str]] = {}
            for alias in self._alias_repository.get_aliases():
                league = cache.setdefault(alias.league_code, {})
                league[self._normalise(alias.raw_name)] = alias.canonical_name
            self._alias_cache = cache
        return self._alias_cache

    def _suggest(self, raw_name: str, canonical_names: Sequence[str]) -> list[str]:
        """Advisory candidates for an admin to accept or reject.

        Compared casefolded — ``difflib`` is case-sensitive, so a shouted feed
        name would otherwise score worse than the same name in title case —
        then mapped back to the canonical spelling.
        """
        if not raw_name.strip() or not canonical_names:
            return []
        by_folded = {self._normalise(name): name for name in canonical_names}
        matches = get_close_matches(
            self._normalise(raw_name),
            list(by_folded),
            n=self.config.suggestion_count,
            cutoff=self.config.suggestion_cutoff,
        )
        return [by_folded[match] for match in matches]

    @staticmethod
    def _normalise(name: str) -> str:
        return name.strip().casefold()


class FixtureNameResolver:
    """Resolves both teams of a fixture, queuing whatever it cannot verify."""

    def __init__(
        self,
        resolver: TeamNameResolver,
        sink: UnresolvedNameSink | None = None,
    ) -> None:
        self._resolver = resolver
        self._sink = sink or NullUnresolvedNameSink()

    def resolve_pair(
        self, league_code: str, home_name: str, away_name: str
    ) -> FixtureResolution:
        home = self._resolver.resolve(
            TeamNameQuery(league_code=league_code, raw_name=home_name)
        )
        away = self._resolver.resolve(
            TeamNameQuery(league_code=league_code, raw_name=away_name)
        )
        unresolved = [r for r in (home, away) if not r.resolved]
        for resolution in unresolved:
            self._record(resolution)
        return FixtureResolution(
            home_team=home.canonical,
            away_team=away.canonical,
            unresolved=unresolved,
        )

    def _record(self, resolution: Resolution) -> None:
        """Recording is best-effort: a broken queue must not block resolution."""
        try:
            self._sink.record(resolution)
        except Exception:
            return None
