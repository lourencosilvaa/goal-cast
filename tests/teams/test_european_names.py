"""Tests for resolving openfootball names against the domestic registry.

openfootball spells clubs in full and inconsistently across seasons — the same
side appears as "Real Madrid" and "Real Madrid CF", "SL Benfica" and "Sport
Lisboa e Benfica". Only 17 of its 382 spellings match this project's
football-data keys exactly, so nearly every name needs a human decision.

What this module does is make that decision *safe* to offer: the country code
narrows the candidates to one country's leagues, so a suggestion is never a
plausible-looking club from the wrong country.
"""

import pandas as pd

from config.config_loader import EuropeanConfig, TeamAliasConfig
from src.teams.european_names import (
    EuropeanName,
    EuropeanNameResolver,
    NameReviewSummary,
)
from src.teams.resolver import TeamAlias, TeamAliasRepository, TeamNameResolver

_REGISTRY = {
    "E0": ["Arsenal", "Man City"],
    "D1": ["FC Koln", "Bayern Munich"],
    "N1": ["Sparta Rotterdam", "Ajax"],
    "P1": ["Sp Lisbon", "Benfica"],
}


class _Aliases(TeamAliasRepository):
    def __init__(self, aliases: list[TeamAlias] | None = None) -> None:
        self._aliases = aliases or []

    def get_aliases(self) -> list[TeamAlias]:
        return self._aliases


def _config(**overrides) -> EuropeanConfig:
    values = dict(
        country_leagues={
            "ENG": ["E0"],
            "GER": ["D1"],
            "NED": ["N1"],
            "POR": ["P1"],
        },
        alias_scope="EU",
    )
    values.update(overrides)
    return EuropeanConfig(**values)


def _resolver(aliases: list[TeamAlias] | None = None, **overrides):
    names = TeamNameResolver(_REGISTRY, _Aliases(aliases), TeamAliasConfig())
    return EuropeanNameResolver(_config(**overrides), names)


class TestQueryConstruction:
    def test_country_selects_its_leagues(self):
        query = _resolver().query_for(
            EuropeanName(raw_name="Arsenal FC", country="ENG", competition="CL")
        )
        assert query.candidate_league_codes == ("E0",)

    def test_a_country_with_several_tiers_offers_all_of_them(self):
        resolver = _resolver(country_leagues={"ENG": ["E0", "E1", "E2"]})
        query = resolver.query_for(
            EuropeanName(raw_name="Arsenal FC", country="ENG", competition="CL")
        )
        assert query.candidate_league_codes == ("E0", "E1", "E2")

    def test_untracked_country_offers_nothing(self):
        query = _resolver().query_for(
            EuropeanName(raw_name="AC Sparta Praha", country="CZE", competition="CL")
        )
        assert query.candidate_league_codes == ()

    def test_missing_country_offers_nothing(self):
        query = _resolver().query_for(
            EuropeanName(raw_name="Some Club", country=None, competition="CL")
        )
        assert query.candidate_league_codes == ()

    def test_alias_scope_is_shared_across_competitions(self):
        """A club approved once must be recognised in every UEFA competition,
        so the scope carries the country but never the competition."""
        champions = _resolver().query_for(
            EuropeanName(raw_name="Arsenal FC", country="ENG", competition="CL")
        )
        conference = _resolver().query_for(
            EuropeanName(raw_name="Arsenal FC", country="ENG", competition="UECL")
        )
        assert champions.alias_scope == conference.alias_scope

    def test_alias_scope_encodes_the_country(self):
        """The review UI recovers the country from the scope, so suggestions
        stay narrowed without adding a column to the pending table."""
        query = _resolver().query_for(
            EuropeanName(raw_name="Arsenal FC", country="ENG", competition="CL")
        )
        assert query.alias_scope == "EU-ENG"

    def test_scope_without_a_country_falls_back_to_the_bare_prefix(self):
        query = _resolver().query_for(
            EuropeanName(raw_name="Some Club", country=None, competition="CL")
        )
        assert query.alias_scope == "EU"

    def test_competition_is_preserved_for_reporting(self):
        query = _resolver().query_for(
            EuropeanName(raw_name="Arsenal FC", country="ENG", competition="UECL")
        )
        assert query.league_code == "UECL"


class TestResolution:
    def test_exact_match_resolves(self):
        result = _resolver().resolve(
            EuropeanName(raw_name="Arsenal", country="ENG", competition="CL")
        )
        assert result.canonical == "Arsenal"
        assert result.status == "exact"

    def test_close_name_is_suggested_not_applied(self):
        result = _resolver().resolve(
            EuropeanName(raw_name="Arsenal FC", country="ENG", competition="CL")
        )
        assert not result.resolved
        assert "Arsenal" in result.suggestions

    def test_suggestions_never_cross_a_border(self):
        """The whole point: a Czech side must not be offered a Dutch one."""
        result = _resolver().resolve(
            EuropeanName(raw_name="AC Sparta Praha", country="CZE", competition="CL")
        )
        assert result.suggestions == []

    def test_german_name_is_matched_against_german_teams(self):
        result = _resolver().resolve(
            EuropeanName(raw_name="1. FC Köln", country="GER", competition="CL")
        )
        assert "FC Koln" in result.suggestions

    def test_approved_alias_resolves(self):
        aliases = [
            TeamAlias(
                league_code="EU-POR",
                raw_name="Sporting Clube de Portugal",
                canonical_name="Sp Lisbon",
            )
        ]
        result = _resolver(aliases).resolve(
            EuropeanName(
                raw_name="Sporting Clube de Portugal",
                country="POR",
                competition="CL",
            )
        )
        assert result.canonical == "Sp Lisbon"


class TestExtractionFromCorpus:
    def _corpus(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Div": ["CL", "CL", "EL"],
                "HomeTeam": ["Arsenal", "Arsenal", "Benfica"],
                "AwayTeam": ["Benfica", "1. FC Köln", "Arsenal"],
                "HomeCountry": ["ENG", "ENG", "POR"],
                "AwayCountry": ["POR", "GER", "ENG"],
            }
        )

    def test_collects_both_sides(self):
        names = _resolver().names_in(self._corpus())
        assert len(names) == 3

    def test_deduplicates_repeated_names(self):
        names = _resolver().names_in(self._corpus())
        assert sorted(n.raw_name for n in names) == [
            "1. FC Köln",
            "Arsenal",
            "Benfica",
        ]

    def test_keeps_the_country_with_the_name(self):
        names = {n.raw_name: n.country for n in _resolver().names_in(self._corpus())}
        assert names["1. FC Köln"] == "GER"
        assert names["Benfica"] == "POR"

    def test_empty_corpus_yields_nothing(self):
        assert _resolver().names_in(pd.DataFrame()) == []


class TestReviewSummary:
    def _corpus(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Div": ["CL", "CL"],
                "HomeTeam": ["Arsenal", "AC Sparta Praha"],
                "AwayTeam": ["Arsenal FC", "FC Astana"],
                "HomeCountry": ["ENG", "CZE"],
                "AwayCountry": ["ENG", "KAZ"],
            }
        )

    def test_splits_names_by_outcome(self):
        summary = _resolver().review(self._corpus())
        assert isinstance(summary, NameReviewSummary)
        assert [r.raw_name for r in summary.resolved] == ["Arsenal"]

    def test_names_with_suggestions_are_reviewable(self):
        summary = _resolver().review(self._corpus())
        assert [r.raw_name for r in summary.reviewable] == ["Arsenal FC"]

    def test_names_without_candidates_are_untrackable(self):
        """Teams from leagues this project does not carry at all."""
        summary = _resolver().review(self._corpus())
        assert sorted(r.raw_name for r in summary.untrackable) == [
            "AC Sparta Praha",
            "FC Astana",
        ]

    def test_counts_add_up(self):
        summary = _resolver().review(self._corpus())
        assert summary.total == 4
        assert (
            len(summary.resolved) + len(summary.reviewable) + len(summary.untrackable)
            == summary.total
        )


class TestScopeRoundTrip:
    """The review UI recovers the country from the stored scope."""

    def test_country_is_recoverable(self):
        resolver = _resolver()
        scope = resolver.alias_scope_for("POR")
        assert resolver.country_in_scope(scope) == "POR"

    def test_bare_scope_carries_no_country(self):
        resolver = _resolver()
        assert resolver.country_in_scope("EU") is None

    def test_a_domestic_league_code_carries_no_country(self):
        assert _resolver().country_in_scope("E0") is None

    def test_an_alias_approved_under_one_competition_serves_another(self):
        aliases = [
            TeamAlias(
                league_code="EU-POR",
                raw_name="Sport Lisboa e Benfica",
                canonical_name="Benfica",
            )
        ]
        resolver = _resolver(aliases)
        for competition in ("CL", "EL", "UECL"):
            result = resolver.resolve(
                EuropeanName(
                    raw_name="Sport Lisboa e Benfica",
                    country="POR",
                    competition=competition,
                )
            )
            assert result.canonical == "Benfica"


class TestBucketingDependsOnCountryNotSimilarity:
    """A tracked country always means reviewable, however poor the match.

    "FC Internazionale Milano" scores nothing against "Inter" and "PSV"
    nothing against "PSV Eindhoven", but both are ordinary sides an admin can
    pick from a list. Filing them as untrackable would hide real work.
    """

    def _corpus(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Div": ["CL", "CL"],
                "HomeTeam": ["FC Internazionale Milano", "AC Sparta Praha"],
                "AwayTeam": ["Wholly Unlike Anything", "FC Astana"],
                "HomeCountry": ["ITA", "CZE"],
                "AwayCountry": ["ENG", "KAZ"],
            }
        )

    def _resolver(self):
        names = TeamNameResolver(
            {"I1": ["Inter", "Milan"], "E0": ["Arsenal"]},
            _Aliases(),
            TeamAliasConfig(),
        )
        return EuropeanNameResolver(
            _config(country_leagues={"ITA": ["I1"], "ENG": ["E0"]}), names
        )

    def test_tracked_country_with_no_suggestion_is_still_reviewable(self):
        summary = self._resolver().review(self._corpus())
        assert "FC Internazionale Milano" in [r.raw_name for r in summary.reviewable]

    def test_a_name_nothing_resembles_is_still_reviewable(self):
        summary = self._resolver().review(self._corpus())
        assert "Wholly Unlike Anything" in [r.raw_name for r in summary.reviewable]

    def test_untracked_country_stays_untrackable(self):
        summary = self._resolver().review(self._corpus())
        assert sorted(r.raw_name for r in summary.untrackable) == [
            "AC Sparta Praha",
            "FC Astana",
        ]
