"""Resolution of openfootball team names against the domestic registry.

openfootball spells clubs in full and, across fifteen seasons, inconsistently:
the same side appears as "Real Madrid" and "Real Madrid CF", "SL Benfica" and
"Sport Lisboa e Benfica". Only 17 of its 382 spellings match this project's
football-data keys exactly, so almost every name needs a human decision — and
the registry itself is not the problem to solve, because football-data's short
keys are what every model, dataset and artefact is already keyed by.

What this module does is make that decision *safe* to put in front of a human.
Searching all 21 leagues at once produces confident nonsense: "AC Sparta
Praha" scores well against "Sparta Rotterdam", and accepting it would attribute
a Czech club's history to a Dutch one. The country code openfootball ships
with every name narrows the candidates to that country, so a suggestion is
never a plausible-looking club from somewhere else, and a country this project
does not track yields no candidates at all — which is the honest answer.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config.config_loader import EuropeanConfig
from src.teams.resolver import Resolution, TeamNameQuery, TeamNameResolver


@dataclass(frozen=True)
class EuropeanName:
    """One team spelling as it appears in a European competition."""

    raw_name: str
    country: Optional[str]
    competition: str


@dataclass
class NameReviewSummary:
    """Where every distinct name stands, split by what a human must do.

    The three buckets are deliberately different kinds of outcome, not
    degrees of confidence:

    * ``resolved`` — matched outright, nothing to do;
    * ``reviewable`` — a real candidate exists, so an admin can decide;
    * ``untrackable`` — the club's country has no league in this project, so
      there is nothing to match against and no decision to make.

    Keeping the third apart matters: those names are not a backlog, and
    presenting them as one would bury the names that *can* be resolved.
    """

    resolved: list[Resolution] = field(default_factory=list)
    reviewable: list[Resolution] = field(default_factory=list)
    untrackable: list[Resolution] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.resolved) + len(self.reviewable) + len(self.untrackable)


class EuropeanNameResolver:
    """Maps openfootball spellings onto canonical football-data names.

    The underlying ``TeamNameResolver`` is injected, so this adds only the
    country-scoping policy and stays testable without a registry file.
    """

    #: Separates the scope prefix from the country, e.g. ``EU-POR``.
    SCOPE_SEPARATOR = "-"

    HOME_COLUMNS = ("HomeTeam", "HomeCountry")
    AWAY_COLUMNS = ("AwayTeam", "AwayCountry")

    def __init__(self, config: EuropeanConfig, resolver: TeamNameResolver) -> None:
        self.config = config
        self._resolver = resolver

    def query_for(self, name: EuropeanName) -> TeamNameQuery:
        """The scoped query for one European name."""
        leagues = self.config.country_leagues.get(name.country or "", [])
        return TeamNameQuery(
            league_code=name.competition,
            raw_name=name.raw_name,
            candidate_league_codes=tuple(leagues),
            alias_scope=self.alias_scope_for(name.country),
        )

    def alias_scope_for(self, country: Optional[str]) -> str:
        """Where an approved alias for a club of this country is stored.

        The scope carries the country but deliberately *not* the competition:
        a side plays the Champions League one year and the Europa League the
        next, and should not need approving twice.

        Encoding the country here rather than adding a column to the pending
        table is what lets the review UI narrow its suggestions to one
        country — the queue only ever stores a scope and a raw name.
        """
        if not country:
            return self.config.alias_scope
        return f"{self.config.alias_scope}{self.SCOPE_SEPARATOR}{country}"

    def country_in_scope(self, scope: str) -> Optional[str]:
        """The country encoded in an alias scope, if it carries one."""
        prefix = f"{self.config.alias_scope}{self.SCOPE_SEPARATOR}"
        if not scope.startswith(prefix):
            return None
        return scope[len(prefix) :] or None

    def resolve(self, name: EuropeanName) -> Resolution:
        """Resolve one name, or report what an admin could choose from."""
        return self._resolver.resolve(self.query_for(name))

    def names_in(self, corpus: pd.DataFrame) -> list[EuropeanName]:
        """Every distinct spelling in a corpus, with its country.

        A club is listed once even though it appears in many matches and
        possibly several competitions — the first competition it was seen in
        is kept purely so an admin has context.
        """
        if corpus is None or corpus.empty:
            return []

        seen: dict[str, EuropeanName] = {}
        for team_column, country_column in (self.HOME_COLUMNS, self.AWAY_COLUMNS):
            if team_column not in corpus.columns:
                continue
            for _, row in corpus.iterrows():
                raw = str(row.get(team_column) or "").strip()
                if not raw or raw in seen:
                    continue
                country = row.get(country_column)
                seen[raw] = EuropeanName(
                    raw_name=raw,
                    country=None if pd.isna(country) else str(country),
                    competition=str(row.get("Div") or ""),
                )
        return list(seen.values())

    def review(self, corpus: pd.DataFrame) -> NameReviewSummary:
        """Resolve every name in a corpus and sort it into the three buckets.

        A name is reviewable whenever its country *has* tracked leagues —
        not merely when a suggestion cleared the similarity cutoff. Those are
        different questions, and conflating them hides real work: "FC
        Internazionale Milano" scores nothing against "Inter" and "PSV"
        nothing against "PSV Eindhoven", yet both are ordinary Serie A and
        Eredivisie sides an admin can pick from a list in seconds.
        """
        summary = NameReviewSummary()
        for name in self.names_in(corpus):
            query = self.query_for(name)
            resolution = self._resolver.resolve(query)
            if resolution.resolved:
                summary.resolved.append(resolution)
            elif query.candidate_league_codes:
                summary.reviewable.append(resolution)
            else:
                summary.untrackable.append(resolution)
        return summary
