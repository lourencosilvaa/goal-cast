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
    """A scraped name to resolve within one competition.

    Domestic fixtures need nothing beyond ``league_code``: the competition a
    name appeared in is also the pool of teams it can belong to. European
    competitions break that identity — a Champions League tie arrives under
    ``CL``, which holds no teams of its own — so two optional fields let the
    caller say where to look without widening the search for everyone else.

    Attributes:
        league_code: The competition the name appeared in. Always what gets
            reported to an admin, whatever was searched.
        raw_name: The spelling to resolve.
        candidate_league_codes: Leagues whose teams are candidates. Empty
            means "just ``league_code``", preserving the domestic behaviour.
            Supplying the leagues of the team's own country is what keeps a
            cross-league search safe — without it, "AC Sparta Praha" is close
            enough to "Sparta Rotterdam" to be accepted by mistake.
        alias_scope: Where approved aliases for this name live, and where an
            unresolved one gets queued. ``None`` means ``league_code``.
            European competitions share one scope so a club approved once is
            recognised in all of them.
        alias_search_scopes: Scopes to *read* aliases from. Empty means "just
            ``alias_scope``". Reading and writing are separate because a
            European fixture carries no country: corpus names are approved
            under ``EU-BEL``, but a fixture can only be queued under plain
            ``EU``. Without a wider read, an already-approved club such as
            "Union Saint-Gilloise" would come back unresolved and need
            approving a second time.
    """

    league_code: str
    raw_name: str
    candidate_league_codes: tuple[str, ...] = ()
    alias_scope: str | None = None
    alias_search_scopes: tuple[str, ...] = ()

    @property
    def search_league_codes(self) -> tuple[str, ...]:
        """The leagues whose teams are candidates for this name."""
        return self.candidate_league_codes or (self.league_code,)

    @property
    def alias_key(self) -> str:
        """The scope an unresolved name is queued under."""
        return self.alias_scope or self.league_code

    @property
    def alias_search_keys(self) -> tuple[str, ...]:
        """The scopes approved aliases are looked up in, in priority order."""
        if not self.alias_search_scopes:
            return (self.alias_key,)
        # The write scope always wins, so a targeted approval cannot be
        # shadowed by an inherited one from a broader scope.
        ordered = [self.alias_key]
        ordered.extend(s for s in self.alias_search_scopes if s != self.alias_key)
        return tuple(ordered)


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
    #: Where an approved alias for this name belongs. Usually the competition
    #: itself, but a cross-league competition scopes by country instead, and
    #: the review queue must store *this* rather than the competition — three
    #: competitions share one club, and the competition alone loses the
    #: country the review screen needs to narrow its suggestions.
    alias_scope: str | None = None

    @property
    def scope(self) -> str:
        """The key an approved alias for this name is stored under."""
        return self.alias_scope or self.league_code

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

    def _candidates(self, query: TeamNameQuery) -> list[str]:
        """Every canonical name the query may resolve to, de-duplicated.

        A league code with no registry entry contributes nothing rather than
        raising: an unknown country simply has no candidates, which is the
        honest answer for a team from a league this project does not track.
        """
        names: list[str] = []
        seen: set[str] = set()
        for code in query.search_league_codes:
            for name in self._canonical.get(code, []):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    def _alias_target(self, query: TeamNameQuery, key: str) -> str | None:
        """The approved canonical name, searching each scope in priority order.

        Multiple scopes exist because a club is approved once, from whichever
        source first met it, but may be seen again from a source that knows
        less about it — a corpus row carries the country, an API fixture does
        not. Searching only the narrower scope would demand a second approval
        for a club already reviewed.
        """
        aliases = self._aliases()
        for scope in query.alias_search_keys:
            target = aliases.get(scope, {}).get(key)
            if target is not None:
                return target
        return None

    def resolve(self, query: TeamNameQuery) -> Resolution:
        """Resolve one scraped name within its competition."""
        canonical_names = self._candidates(query)
        key = self._normalise(query.raw_name)
        if not key or not canonical_names:
            return Resolution(
                raw_name=query.raw_name,
                league_code=query.league_code,
                alias_scope=query.alias_key,
                suggestions=self._suggest(query.raw_name, canonical_names),
            )

        for name in canonical_names:
            if self._normalise(name) == key:
                return Resolution(
                    raw_name=query.raw_name,
                    league_code=query.league_code,
                    alias_scope=query.alias_key,
                    canonical=name,
                    status=Resolution.STATUS_EXACT,
                )

        alias_target = self._alias_target(query, key)
        # A stale alias must never inject a team the data set does not have.
        if alias_target is not None and alias_target in canonical_names:
            return Resolution(
                raw_name=query.raw_name,
                league_code=query.league_code,
                alias_scope=query.alias_key,
                canonical=alias_target,
                status=Resolution.STATUS_ALIAS,
            )

        return Resolution(
            raw_name=query.raw_name,
            league_code=query.league_code,
            alias_scope=query.alias_key,
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
