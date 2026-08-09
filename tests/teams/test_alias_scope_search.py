"""An approved alias must count wherever the club appears.

Corpus names arrive with a country and are approved under a scope like
``EU-BEL``. Fixture names arrive from an API with no country at all, so they
can only be looked up under the plain ``EU`` scope. Unless the read side can
span both, the same 99 clubs would have to be approved a second time — and
"Union Saint-Gilloise → St. Gilloise", already approved from the corpus, would
come back as unresolved the moment it appeared in an actual fixture.

So a query separates the scope it *writes* to from the scopes it *reads*,
exactly as it already separates ``league_code`` from ``candidate_league_codes``.
"""

from config.config_loader import TeamAliasConfig
from src.teams.resolver import (
    TeamAlias,
    TeamAliasRepository,
    TeamNameQuery,
    TeamNameResolver,
)

_REGISTRY = {"B1": ["St. Gilloise", "Anderlecht"], "G1": ["Olympiakos"]}


class _Aliases(TeamAliasRepository):
    def __init__(self, aliases) -> None:
        self._aliases = aliases

    def get_aliases(self):
        return list(self._aliases)


_APPROVED = [
    TeamAlias("EU-BEL", "Union Saint-Gilloise", "St. Gilloise"),
    TeamAlias("EU-GRE", "Olympiakos Piraeus", "Olympiakos"),
]


def _resolver() -> TeamNameResolver:
    return TeamNameResolver(_REGISTRY, _Aliases(_APPROVED), TeamAliasConfig())


class TestSearchScopes:
    def test_an_alias_from_another_scope_resolves_when_searched(self):
        query = TeamNameQuery(
            league_code="CLQ",
            raw_name="Union Saint-Gilloise",
            candidate_league_codes=("B1", "G1"),
            alias_scope="EU",
            alias_search_scopes=("EU", "EU-BEL", "EU-GRE"),
        )
        assert _resolver().resolve(query).canonical == "St. Gilloise"

    def test_each_searched_scope_is_consulted(self):
        query = TeamNameQuery(
            league_code="CLQ",
            raw_name="Olympiakos Piraeus",
            candidate_league_codes=("B1", "G1"),
            alias_scope="EU",
            alias_search_scopes=("EU", "EU-BEL", "EU-GRE"),
        )
        assert _resolver().resolve(query).canonical == "Olympiakos"

    def test_without_the_wider_search_it_stays_unresolved(self):
        """The bug this fixes, pinned so it cannot silently return."""
        query = TeamNameQuery(
            league_code="CLQ",
            raw_name="Union Saint-Gilloise",
            candidate_league_codes=("B1",),
            alias_scope="EU",
        )
        assert _resolver().resolve(query).canonical is None

    def test_the_write_scope_is_unchanged_by_widening_the_search(self):
        """Queued names must still land in one predictable place."""
        query = TeamNameQuery(
            league_code="CLQ",
            raw_name="Totally Unknown FC",
            candidate_league_codes=("B1",),
            alias_scope="EU",
            alias_search_scopes=("EU", "EU-BEL"),
        )
        assert _resolver().resolve(query).scope == "EU"

    def test_search_scopes_default_to_the_write_scope(self):
        """Existing callers keep exactly their current behaviour."""
        query = TeamNameQuery(
            league_code="CLQ", raw_name="x", alias_scope="EU-BEL"
        )
        assert query.alias_search_keys == ("EU-BEL",)

    def test_a_domestic_query_is_untouched(self):
        query = TeamNameQuery(league_code="B1", raw_name="x")
        assert query.alias_search_keys == ("B1",)

    def test_an_alias_still_cannot_inject_an_unavailable_team(self):
        """Widening the search must not weaken the stale-alias guard."""
        query = TeamNameQuery(
            league_code="CLQ",
            raw_name="Olympiakos Piraeus",
            candidate_league_codes=("B1",),  # Greek league not searched
            alias_scope="EU",
            alias_search_scopes=("EU", "EU-GRE"),
        )
        assert _resolver().resolve(query).canonical is None
