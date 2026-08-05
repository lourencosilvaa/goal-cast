"""Tests for canonical team-name resolution.

Fixtures scraped from FlashScore carry that site's spelling ("Sporting CP"),
while every model and data set in this project is keyed by the football-data
spelling ("Sp Lisbon"). Resolution turns the former into the latter — and,
crucially, refuses to guess: anything not matched exactly or by an
**admin-approved** alias comes back unresolved with advisory suggestions.
"""

import pytest

from config.config_loader import TeamAliasConfig
from src.teams.resolver import (
    FixtureNameResolver,
    ListUnresolvedNameSink,
    Resolution,
    TeamAlias,
    TeamAliasRepository,
    TeamNameQuery,
    TeamNameResolver,
)

CANONICAL = {
    "P1": ["Sp Lisbon", "Porto", "Benfica", "Sp Braga"],
    "E0": ["Arsenal", "Man City", "Man United", "Chelsea"],
}

#: Mirrors the shipped defaults. Set explicitly so the expectations below do
#: not silently change meaning if the YAML is retuned (§7.3).
CONFIG = TeamAliasConfig(
    seed_path="config/team_aliases.yaml", suggestion_count=5, suggestion_cutoff=0.4
)


class FakeAliasRepository(TeamAliasRepository):
    """In-memory alias source holding only approved mappings."""

    def __init__(self, aliases: list[TeamAlias] | None = None) -> None:
        self._aliases = list(aliases or [])

    def get_aliases(self) -> list[TeamAlias]:
        return list(self._aliases)


def resolver(aliases: list[TeamAlias] | None = None) -> TeamNameResolver:
    return TeamNameResolver(
        canonical_teams=CANONICAL,
        alias_repository=FakeAliasRepository(aliases),
        config=CONFIG,
    )


def query(raw_name: str, league_code: str = "P1") -> TeamNameQuery:
    return TeamNameQuery(league_code=league_code, raw_name=raw_name)


class TestExactMatches:

    def test_canonical_name_resolves_to_itself(self):
        result = resolver().resolve(query("Sp Lisbon"))
        assert result.canonical == "Sp Lisbon"
        assert result.status == Resolution.STATUS_EXACT
        assert result.resolved is True

    def test_matching_ignores_case(self):
        assert resolver().resolve(query("sp lisbon")).canonical == "Sp Lisbon"

    def test_matching_ignores_surrounding_whitespace(self):
        assert resolver().resolve(query("  Porto  ")).canonical == "Porto"

    def test_exact_match_needs_no_alias(self):
        result = resolver().resolve(query("Porto"))
        assert result.status == Resolution.STATUS_EXACT

    def test_matching_is_scoped_to_the_league(self):
        """Arsenal exists — but not in Liga Portugal."""
        result = resolver().resolve(query("Arsenal", league_code="P1"))
        assert result.resolved is False


class TestApprovedAliases:

    @staticmethod
    def _aliases() -> list[TeamAlias]:
        return [
            TeamAlias(league_code="P1", raw_name="Sporting CP", canonical_name="Sp Lisbon"),
            TeamAlias(league_code="E0", raw_name="Manchester City", canonical_name="Man City"),
        ]

    def test_alias_resolves_to_the_canonical_name(self):
        result = resolver(self._aliases()).resolve(query("Sporting CP"))
        assert result.canonical == "Sp Lisbon"
        assert result.status == Resolution.STATUS_ALIAS
        assert result.resolved is True

    def test_alias_matching_ignores_case_and_whitespace(self):
        result = resolver(self._aliases()).resolve(query(" sporting cp "))
        assert result.canonical == "Sp Lisbon"

    def test_alias_is_scoped_to_its_league(self):
        """A Premier League alias must not resolve a Liga Portugal name."""
        result = resolver(self._aliases()).resolve(
            query("Manchester City", league_code="P1")
        )
        assert result.resolved is False

    def test_alias_for_the_right_league_resolves(self):
        result = resolver(self._aliases()).resolve(
            query("Manchester City", league_code="E0")
        )
        assert result.canonical == "Man City"

    def test_alias_pointing_at_an_unknown_canonical_name_is_rejected(self):
        """A stale alias must not inject a name the data set does not have."""
        stale = [TeamAlias(league_code="P1", raw_name="Sporting CP", canonical_name="Gone FC")]
        result = resolver(stale).resolve(query("Sporting CP"))
        assert result.resolved is False
        assert result.canonical is None


class TestUnresolvedNames:

    def test_unknown_name_is_unresolved(self):
        result = resolver().resolve(query("Totally Unknown"))
        assert result.canonical is None
        assert result.status == Resolution.STATUS_UNRESOLVED
        assert result.resolved is False

    def test_unresolved_name_carries_suggestions(self):
        result = resolver().resolve(query("Sporting"))
        assert "Sp Lisbon" in result.suggestions

    def test_suggestions_are_capped_by_config(self):
        config = TeamAliasConfig(
            seed_path="config/team_aliases.yaml",
            suggestion_count=1,
            suggestion_cutoff=0.1,
        )
        result = TeamNameResolver(
            canonical_teams=CANONICAL,
            alias_repository=FakeAliasRepository(),
            config=config,
        ).resolve(query("Sporting"))
        assert len(result.suggestions) == 1

    def test_a_high_cutoff_suppresses_weak_suggestions(self):
        config = TeamAliasConfig(
            seed_path="config/team_aliases.yaml",
            suggestion_count=3,
            suggestion_cutoff=0.99,
        )
        result = TeamNameResolver(
            canonical_teams=CANONICAL,
            alias_repository=FakeAliasRepository(),
            config=config,
        ).resolve(query("Sporting"))
        assert result.suggestions == []

    def test_suggestions_come_from_the_queried_league_only(self):
        result = resolver().resolve(query("Manchester City", league_code="P1"))
        assert all(name in CANONICAL["P1"] for name in result.suggestions)

    def test_unknown_league_resolves_nothing(self):
        result = resolver().resolve(query("Sp Lisbon", league_code="ZZ"))
        assert result.resolved is False
        assert result.suggestions == []

    def test_empty_name_is_unresolved(self):
        assert resolver().resolve(query("")).resolved is False


class TestFixtureNameResolver:
    """Resolves both sides of a fixture and reports what it could not verify."""

    @staticmethod
    def _aliases() -> list[TeamAlias]:
        return [
            TeamAlias(league_code="P1", raw_name="Sporting CP", canonical_name="Sp Lisbon")
        ]

    def test_both_names_resolved(self):
        fixture = FixtureNameResolver(resolver(self._aliases())).resolve_pair(
            league_code="P1", home_name="Sporting CP", away_name="Porto"
        )
        assert fixture.resolved is True
        assert fixture.home_team == "Sp Lisbon"
        assert fixture.away_team == "Porto"
        assert fixture.unresolved == []

    def test_one_unresolved_name_fails_the_pair(self):
        fixture = FixtureNameResolver(resolver()).resolve_pair(
            league_code="P1", home_name="Sporting CP", away_name="Porto"
        )
        assert fixture.resolved is False
        assert fixture.home_team is None

    def test_unresolved_names_are_reported(self):
        fixture = FixtureNameResolver(resolver()).resolve_pair(
            league_code="P1", home_name="Sporting CP", away_name="Unknown FC"
        )
        assert [r.raw_name for r in fixture.unresolved] == ["Sporting CP", "Unknown FC"]

    def test_unresolved_names_are_sent_to_the_sink(self):
        sink = ListUnresolvedNameSink()
        FixtureNameResolver(resolver(), sink=sink).resolve_pair(
            league_code="P1", home_name="Sporting CP", away_name="Porto"
        )
        assert [r.raw_name for r in sink.recorded] == ["Sporting CP"]

    def test_resolved_pairs_record_nothing(self):
        sink = ListUnresolvedNameSink()
        FixtureNameResolver(resolver(self._aliases()), sink=sink).resolve_pair(
            league_code="P1", home_name="Sporting CP", away_name="Porto"
        )
        assert sink.recorded == []

    def test_a_failing_sink_never_breaks_resolution(self):
        """Recording is best-effort telemetry, not part of the answer."""

        class BrokenSink(ListUnresolvedNameSink):
            def record(self, resolution: Resolution) -> None:
                raise RuntimeError("supabase down")

        fixture = FixtureNameResolver(resolver(), sink=BrokenSink()).resolve_pair(
            league_code="P1", home_name="Unknown FC", away_name="Porto"
        )
        assert fixture.resolved is False


class TestRepositoryInterface:

    def test_fake_implements_the_abstraction(self):
        assert isinstance(FakeAliasRepository(), TeamAliasRepository)

    def test_abstraction_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            TeamAliasRepository()  # type: ignore[abstract]
