"""Tests for resolving names that appear in cross-league competitions.

A Champions League fixture arrives under ``CL``, which is not a league in the
registry — so the existing per-league lookup finds no candidates at all and
every name comes back unresolved with no suggestions.

Widening the search to *every* league would be worse than useless: across 21
leagues the near-misses are actively dangerous. "AC Sparta Praha" scores well
against "Sparta Rotterdam", and accepting that would silently attribute a
Czech side's history to a Dutch one.

The country code openfootball ships with every name is what makes this safe.
It narrows the candidates to the leagues of that country, so a Czech team has
no candidates at all — correctly unresolved — while ``1. FC Köln (GER)`` is
compared only against German sides.
"""

from config.config_loader import TeamAliasConfig
from src.teams.resolver import (
    StaticTeamAliasRepository,
    TeamAlias,
    TeamAliasRepository,
    TeamNameQuery,
    TeamNameResolver,
)

_REGISTRY = {
    "E0": ["Arsenal", "Man City", "Liverpool"],
    "D1": ["FC Koln", "Bayern Munich", "Hoffenheim"],
    "N1": ["Sparta Rotterdam", "Ajax", "PSV Eindhoven"],
    "P1": ["Sp Lisbon", "Benfica", "Porto"],
}


class _Aliases(TeamAliasRepository):
    def __init__(self, aliases: list[TeamAlias] | None = None) -> None:
        self._aliases = aliases or []

    def get_aliases(self) -> list[TeamAlias]:
        return self._aliases


def _resolver(aliases: list[TeamAlias] | None = None) -> TeamNameResolver:
    return TeamNameResolver(_REGISTRY, _Aliases(aliases), TeamAliasConfig())


class TestExistingBehaviourUnchanged:
    """Every current caller passes no candidate scope and must be unaffected."""

    def test_exact_match_within_a_league(self):
        result = _resolver().resolve(TeamNameQuery(league_code="E0", raw_name="Arsenal"))
        assert result.canonical == "Arsenal"

    def test_unknown_league_stays_unresolved(self):
        result = _resolver().resolve(TeamNameQuery(league_code="CL", raw_name="Arsenal"))
        assert not result.resolved
        assert result.suggestions == []

    def test_alias_within_a_league(self):
        aliases = [TeamAlias(league_code="E0", raw_name="Arsenal FC", canonical_name="Arsenal")]
        result = _resolver(aliases).resolve(
            TeamNameQuery(league_code="E0", raw_name="Arsenal FC")
        )
        assert result.canonical == "Arsenal"


class TestCandidateScoping:
    def test_resolves_against_the_supplied_leagues(self):
        result = _resolver().resolve(
            TeamNameQuery(
                league_code="CL",
                raw_name="Arsenal",
                candidate_league_codes=("E0", "E1"),
            )
        )
        assert result.canonical == "Arsenal"
        assert result.status == "exact"

    def test_reports_the_competition_it_appeared_in(self):
        """The competition is what an admin sees, even when candidates differ."""
        result = _resolver().resolve(
            TeamNameQuery(
                league_code="CL", raw_name="Arsenal", candidate_league_codes=("E0",)
            )
        )
        assert result.league_code == "CL"

    def test_suggestions_come_only_from_the_supplied_leagues(self):
        result = _resolver().resolve(
            TeamNameQuery(
                league_code="CL",
                raw_name="1. FC Köln",
                candidate_league_codes=("D1", "D2"),
            )
        )
        assert result.suggestions
        assert all(name in _REGISTRY["D1"] for name in result.suggestions)

    def test_a_dangerous_near_miss_is_ruled_out_by_country(self):
        """'AC Sparta Praha' must never be offered 'Sparta Rotterdam'."""
        result = _resolver().resolve(
            TeamNameQuery(
                league_code="CL",
                raw_name="AC Sparta Praha",
                candidate_league_codes=(),
            )
        )
        assert not result.resolved
        assert result.suggestions == []

    def test_untracked_country_yields_no_candidates(self):
        result = _resolver().resolve(
            TeamNameQuery(
                league_code="CL",
                raw_name="FC Astana",
                candidate_league_codes=(),
            )
        )
        assert not result.resolved

    def test_candidates_from_several_leagues_are_pooled(self):
        result = _resolver().resolve(
            TeamNameQuery(
                league_code="CL",
                raw_name="Benfica",
                candidate_league_codes=("P1", "E0"),
            )
        )
        assert result.canonical == "Benfica"

    def test_unknown_candidate_league_is_ignored(self):
        result = _resolver().resolve(
            TeamNameQuery(
                league_code="CL",
                raw_name="Arsenal",
                candidate_league_codes=("E0", "ZZ"),
            )
        )
        assert result.canonical == "Arsenal"


class TestAliasScope:
    """One approved alias should serve every European competition.

    A club plays the Champions League one year and the Europa League the next.
    Keying its alias to the competition would force an admin to approve the
    same mapping three times.
    """

    def test_alias_resolves_under_a_shared_scope(self):
        aliases = [
            TeamAlias(
                league_code="EU",
                raw_name="Sporting Clube de Portugal",
                canonical_name="Sp Lisbon",
            )
        ]
        result = _resolver(aliases).resolve(
            TeamNameQuery(
                league_code="CL",
                raw_name="Sporting Clube de Portugal",
                candidate_league_codes=("P1",),
                alias_scope="EU",
            )
        )
        assert result.canonical == "Sp Lisbon"
        assert result.status == "alias"

    def test_the_same_alias_serves_another_competition(self):
        aliases = [
            TeamAlias(
                league_code="EU",
                raw_name="Sporting Clube de Portugal",
                canonical_name="Sp Lisbon",
            )
        ]
        result = _resolver(aliases).resolve(
            TeamNameQuery(
                league_code="UECL",
                raw_name="Sporting Clube de Portugal",
                candidate_league_codes=("P1",),
                alias_scope="EU",
            )
        )
        assert result.canonical == "Sp Lisbon"

    def test_alias_pointing_outside_the_candidates_is_refused(self):
        """A stale alias must never inject a team the candidates do not hold."""
        aliases = [
            TeamAlias(league_code="EU", raw_name="Ajax Amsterdam", canonical_name="Ajax")
        ]
        result = _resolver(aliases).resolve(
            TeamNameQuery(
                league_code="CL",
                raw_name="Ajax Amsterdam",
                candidate_league_codes=("P1",),
                alias_scope="EU",
            )
        )
        assert not result.resolved

    def test_scope_defaults_to_the_competition_code(self):
        aliases = [
            TeamAlias(league_code="CL", raw_name="Arsenal FC", canonical_name="Arsenal")
        ]
        result = _resolver(aliases).resolve(
            TeamNameQuery(
                league_code="CL",
                raw_name="Arsenal FC",
                candidate_league_codes=("E0",),
            )
        )
        assert result.canonical == "Arsenal"


class TestStaticSeedStillLoads:
    def test_seed_file_is_read(self, tmp_path):
        seed = tmp_path / "aliases.yaml"
        seed.write_text(
            'aliases:\n  EU:\n    "Sporting Clube de Portugal": "Sp Lisbon"\n',
            encoding="utf-8",
        )
        aliases = StaticTeamAliasRepository(seed).get_aliases()
        assert aliases[0].league_code == "EU"
        assert aliases[0].canonical_name == "Sp Lisbon"


class TestResolutionCarriesItsScope:
    """The queue must store where an approved alias will be looked up.

    ``league_code`` is the competition a name appeared in — right for telling
    an admin what they are looking at, wrong as a storage key, because three
    competitions share one club. Without the scope travelling with the
    resolution the country is lost on the way to the queue, and the review
    screen can no longer narrow its suggestions.
    """

    def test_scope_defaults_to_the_competition(self):
        result = _resolver().resolve(
            TeamNameQuery(league_code="E0", raw_name="Unknown FC")
        )
        assert result.alias_scope == "E0"

    def test_scope_is_carried_from_the_query(self):
        result = _resolver().resolve(
            TeamNameQuery(
                league_code="CL",
                raw_name="Sporting Clube de Portugal",
                candidate_league_codes=("P1",),
                alias_scope="EU-POR",
            )
        )
        assert result.alias_scope == "EU-POR"

    def test_competition_is_still_reported_separately(self):
        result = _resolver().resolve(
            TeamNameQuery(
                league_code="CL",
                raw_name="Sporting Clube de Portugal",
                candidate_league_codes=("P1",),
                alias_scope="EU-POR",
            )
        )
        assert result.league_code == "CL"
        assert result.alias_scope != result.league_code

    def test_resolved_names_carry_it_too(self):
        result = _resolver().resolve(
            TeamNameQuery(
                league_code="CL",
                raw_name="Arsenal",
                candidate_league_codes=("E0",),
                alias_scope="EU-ENG",
            )
        )
        assert result.alias_scope == "EU-ENG"
