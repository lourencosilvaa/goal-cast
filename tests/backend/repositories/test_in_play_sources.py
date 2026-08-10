"""Adapters that plug the app's existing services into the in-play board.

The board (:mod:`src.backend.services.in_play_board`) is written against two
narrow interfaces — "give me a league-day's prediction rows" and "resolve this
spelling" — precisely so it does not import Supabase or the alias registry.
These are the two implementations that do, and the only behaviour worth
pinning is the translation: what gets asked of the wrapped service, and what
comes back when it cannot answer.
"""

from src.backend.repositories.in_play_sources import (
    PredictionServiceSource,
    ResolverNameMatcher,
)
from src.teams.resolver import (
    ChainedTeamAliasRepository,
    StaticTeamAliasRepository,
    TeamAlias,
    TeamAliasRepository,
    TeamNameResolver,
)
from config.config_loader import TeamAliasConfig

_ALIAS_CONFIG = TeamAliasConfig(
    seed_path="config/does-not-exist.yaml",
    suggestion_count=3,
    suggestion_cutoff=0.4,
)


class _LeaguePredictions:
    def __init__(self, matches):
        self.matches = matches


class _Service:
    def __init__(self, matches=None):
        self._matches = matches if matches is not None else [{"home_team": "Porto"}]
        self.calls: list[tuple[str, str]] = []

    def get_league_predictions(self, league_code, target_date=None):
        self.calls.append((league_code, target_date))
        return _LeaguePredictions(self._matches)


class _Aliases(TeamAliasRepository):
    def __init__(self, aliases=()):
        self._aliases = list(aliases)

    def get_aliases(self):
        return self._aliases


def _matcher(aliases=()) -> ResolverNameMatcher:
    return ResolverNameMatcher(
        TeamNameResolver(
            canonical_teams={"P1": ["Porto", "Sp Lisbon"], "E0": ["Arsenal"]},
            alias_repository=_Aliases(aliases),
            config=_ALIAS_CONFIG,
        )
    )


class TestPredictionServiceSource:
    def test_it_returns_the_stored_rows(self):
        source = PredictionServiceSource(_Service([{"home_team": "Braga"}]))
        assert source.matches("P1", "09/08/2026") == [{"home_team": "Braga"}]

    def test_the_day_is_passed_through_not_defaulted(self):
        """The board asks for the day the match kicked off; letting the
        service fall back to "today" is the bug this parameter exists for."""
        service = _Service()
        PredictionServiceSource(service).matches("P1", "09/08/2026")
        assert service.calls == [("P1", "09/08/2026")]

    def test_a_league_with_nothing_stored_is_empty_not_an_error(self):
        assert PredictionServiceSource(_Service([])).matches("P1", "09/08/2026") == []


class TestResolverNameMatcher:
    def test_an_exact_name_resolves_to_itself(self):
        assert _matcher().canonical("P1", "Porto") == "Porto"

    def test_matching_ignores_case_and_padding(self):
        assert _matcher().canonical("P1", "  porto ") == "Porto"

    def test_an_approved_alias_resolves(self):
        aliases = [
            TeamAlias(league_code="P1", raw_name="Sporting CP", canonical_name="Sp Lisbon")
        ]
        assert _matcher(aliases).canonical("P1", "Sporting CP") == "Sp Lisbon"

    def test_an_unknown_name_is_none_never_a_guess(self):
        """"Sporting CP" scores close enough to "Sp Lisbon" to be suggested to
        an admin, and that is exactly why it must not resolve on its own."""
        assert _matcher().canonical("P1", "Sporting CP") is None

    def test_a_name_from_another_league_does_not_resolve(self):
        """The competition is also the pool of teams a name can belong to."""
        assert _matcher().canonical("P1", "Arsenal") is None

    def test_a_league_with_no_registry_entry_resolves_nothing(self):
        assert _matcher().canonical("ZZ", "Porto") is None

    def test_a_broken_alias_source_does_not_take_resolution_down(self):
        """A missing seed file leaves exact matching working."""
        matcher = ResolverNameMatcher(
            TeamNameResolver(
                canonical_teams={"P1": ["Porto"]},
                alias_repository=ChainedTeamAliasRepository(
                    [StaticTeamAliasRepository("config/does-not-exist.yaml")]
                ),
                config=_ALIAS_CONFIG,
            )
        )
        assert matcher.canonical("P1", "Porto") == "Porto"
