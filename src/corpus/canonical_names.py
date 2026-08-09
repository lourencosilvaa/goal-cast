"""Rewrites European rows into the canonical team keys the models use.

This is the step that turns an approved alias into an effect. ELO is keyed by
the team-name string: until "Sport Lisboa e Benfica" becomes "Benfica", the
European result updates a *separate* rating and the domestic pools it exists to
connect stay exactly as disconnected as before. The calibration would run
cleanly and achieve nothing — the worst kind of failure, because it looks like
success.

Untranslatable rows are kept rather than dropped. A club from a league this
project does not carry has no canonical key and never will, but its matches
still carry information: they let that club build a rating of its own from
European play, which is what makes a Benfica-vs-Shakhtar result informative
rather than noise.
"""

from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd

from src.teams.european_names import EuropeanNameResolver


@dataclass
class TranslationReport:
    """What the translation achieved, and what is still missing.

    ``linkable`` is the actionable part: names whose country *is* tracked, so
    an approval would connect them. Names from untracked countries are absent
    from it by design — they are not a backlog, and listing them would bury
    the ones a human can act on.
    """

    #: Distinct names mapped onto a canonical key.
    translated: int = 0
    #: Distinct names left as they were.
    untranslated: int = 0
    #: Untranslated names an approval could still link, worst first.
    linkable: list[str] = field(default_factory=list)
    #: Match appearances those names account for — what the ratings lose.
    unlinked_appearances: int = 0
    #: The mapping actually applied, for reporting and debugging.
    mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class TranslationResult:
    """The rewritten corpus and an account of what happened to it."""

    frame: pd.DataFrame
    report: TranslationReport


class CanonicalCorpusTranslator:
    """Maps a European corpus onto canonical team keys.

    The resolver is injected, so the country-scoping policy lives in one place
    and this stays testable without a registry file.
    """

    HOME_COLUMNS: ClassVar[tuple[str, str]] = ("HomeTeam", "HomeCountry")
    AWAY_COLUMNS: ClassVar[tuple[str, str]] = ("AwayTeam", "AwayCountry")

    def __init__(self, resolver: EuropeanNameResolver) -> None:
        self._resolver = resolver

    def translate(self, corpus: pd.DataFrame) -> TranslationResult:
        """Rewrite team names in place of their canonical keys."""
        if corpus is None or corpus.empty:
            return TranslationResult(
                frame=corpus if corpus is not None else pd.DataFrame(),
                report=TranslationReport(),
            )

        mapping, linkable = self._build_mapping(corpus)
        frame = corpus.copy()
        for team_column, _ in (self.HOME_COLUMNS, self.AWAY_COLUMNS):
            frame[team_column] = frame[team_column].map(lambda n: mapping.get(n, n))

        appearances = self._appearances(corpus, linkable)
        return TranslationResult(
            frame=frame,
            report=TranslationReport(
                translated=len(mapping),
                untranslated=len(self._resolver.names_in(corpus)) - len(mapping),
                linkable=sorted(linkable, key=lambda n: (-appearances.get(n, 0), n)),
                unlinked_appearances=sum(appearances.values()),
                mapping=mapping,
            ),
        )

    def _build_mapping(self, corpus: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
        """Canonical key per name, plus the names an approval could still link."""
        mapping: dict[str, str] = {}
        linkable: list[str] = []
        for name in self._resolver.names_in(corpus):
            query = self._resolver.query_for(name)
            resolution = self._resolver.resolve(name)
            if resolution.resolved and resolution.canonical:
                mapping[name.raw_name] = resolution.canonical
            elif query.candidate_league_codes:
                # The country is tracked, so a canonical key exists to be
                # chosen — this one is simply awaiting a human decision.
                linkable.append(name.raw_name)
        return mapping, linkable

    def _appearances(self, corpus: pd.DataFrame, names: list[str]) -> dict[str, int]:
        """How many match sides each unlinked name accounts for."""
        if not names:
            return {}
        wanted = set(names)
        counts: dict[str, int] = {}
        for team_column, _ in (self.HOME_COLUMNS, self.AWAY_COLUMNS):
            for value in corpus[team_column]:
                if value in wanted:
                    counts[value] = counts.get(value, 0) + 1
        return counts


def build_translator(config: object) -> CanonicalCorpusTranslator:
    """Wire a translator from the shipped registries and approved aliases.

    Alias resolution matches against the *historical* registry: a club
    relegated out of a tracked division keeps years of history the model still
    knows, and must stay matchable to it.
    """
    from src.backend.repositories.team_alias_repository import (
        SupabaseTeamAliasRepository,
    )
    from src.teams.registry import load_team_registry
    from src.teams.resolver import (
        ChainedTeamAliasRepository,
        StaticTeamAliasRepository,
        TeamNameResolver,
    )

    teams_config = config.teams  # type: ignore[attr-defined]
    sources: list[object] = [StaticTeamAliasRepository(teams_config.aliases.seed_path)]
    try:
        from src.backend.core.supabase_client import get_supabase_client
        from src.backend.services.team_alias_service import TeamAliasService

        sources.append(
            SupabaseTeamAliasRepository(TeamAliasService(get_supabase_client()))
        )
    except Exception:
        # Approvals live in Supabase, but a build without it should still run
        # on the committed seed rather than fail outright.
        pass

    names = TeamNameResolver(
        load_team_registry(teams_config.historical_registry_path),
        ChainedTeamAliasRepository(sources),  # type: ignore[arg-type]
        teams_config.aliases,
    )
    return CanonicalCorpusTranslator(
        EuropeanNameResolver(config.european, names)  # type: ignore[attr-defined]
    )
