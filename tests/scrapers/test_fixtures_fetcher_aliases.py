"""Tests for canonical-name resolution of FlashScore fixtures.

FlashScore is the last-resort fixture source and spells teams its own way. Its
names used to reach the model untranslated, where an unmatched name silently
produced a league-average prediction. Now a fixture is emitted only when both
teams resolve; anything else is dropped and queued for admin review.
"""

from unittest.mock import MagicMock

from config.config_loader import TeamAliasConfig
from src.scrapers.fixtures_fetcher import Fixture, resolve_fixture_names
from src.teams.resolver import (
    FixtureNameResolver,
    ListUnresolvedNameSink,
    TeamAlias,
    TeamAliasRepository,
    TeamNameResolver,
)

CANONICAL = {"P1": ["Sp Lisbon", "Porto", "Benfica"]}
CONFIG = TeamAliasConfig(
    seed_path="config/team_aliases.yaml", suggestion_count=5, suggestion_cutoff=0.4
)


class FakeAliasRepository(TeamAliasRepository):
    def __init__(self, aliases: list[TeamAlias] | None = None) -> None:
        self._aliases = list(aliases or [])

    def get_aliases(self) -> list[TeamAlias]:
        return list(self._aliases)


def fixture(home: str, away: str, division: str = "P1") -> Fixture:
    return Fixture(
        division=division,
        league="Liga Portugal",
        date="01/05/2026",
        time="20:00",
        home_team=home,
        away_team=away,
        b365_home=None,
        b365_draw=None,
        b365_away=None,
    )


def fixture_resolver(
    aliases: list[TeamAlias] | None = None, sink=None
) -> FixtureNameResolver:
    return FixtureNameResolver(
        TeamNameResolver(
            canonical_teams=CANONICAL,
            alias_repository=FakeAliasRepository(aliases),
            config=CONFIG,
        ),
        sink=sink,
    )


APPROVED = [TeamAlias("P1", "Sporting CP", "Sp Lisbon")]


class TestResolveFixtureNames:

    def test_canonical_fixtures_pass_through(self):
        fixtures = [fixture("Sp Lisbon", "Porto")]
        assert resolve_fixture_names(fixtures, fixture_resolver()) == fixtures

    def test_approved_alias_is_applied(self):
        result = resolve_fixture_names(
            [fixture("Sporting CP", "Porto")], fixture_resolver(APPROVED)
        )
        assert len(result) == 1
        assert result[0].home_team == "Sp Lisbon"
        assert result[0].away_team == "Porto"

    def test_other_fixture_fields_are_preserved(self):
        original = fixture("Sporting CP", "Porto")
        resolved = resolve_fixture_names([original], fixture_resolver(APPROVED))[0]
        assert resolved.division == original.division
        assert resolved.date == original.date
        assert resolved.time == original.time
        assert resolved.league == original.league

    def test_odds_survive_resolution(self):
        original = fixture("Sporting CP", "Porto")
        priced = Fixture(
            **{
                **original.__dict__,
                "b365_home": 1.8,
                "b365_draw": 3.5,
                "b365_away": 4.2,
            }
        )
        resolved = resolve_fixture_names([priced], fixture_resolver(APPROVED))[0]
        assert resolved.has_odds is True
        assert resolved.b365_home == 1.8

    def test_unresolved_fixture_is_dropped(self):
        """The old behaviour predicted this from league averages."""
        result = resolve_fixture_names(
            [fixture("Sporting CP", "Porto")], fixture_resolver()
        )
        assert result == []

    def test_one_bad_name_drops_only_its_own_fixture(self):
        fixtures = [fixture("Sporting CP", "Porto"), fixture("Benfica", "Porto")]
        result = resolve_fixture_names(fixtures, fixture_resolver())
        assert [f.home_team for f in result] == ["Benfica"]

    def test_unresolved_names_are_queued_for_review(self):
        sink = ListUnresolvedNameSink()
        resolve_fixture_names(
            [fixture("Sporting CP", "Unknown FC")], fixture_resolver(sink=sink)
        )
        assert {r.raw_name for r in sink.recorded} == {"Sporting CP", "Unknown FC"}

    def test_resolved_fixtures_queue_nothing(self):
        sink = ListUnresolvedNameSink()
        resolve_fixture_names(
            [fixture("Sp Lisbon", "Porto")], fixture_resolver(sink=sink)
        )
        assert sink.recorded == []

    def test_empty_input_yields_empty_output(self):
        assert resolve_fixture_names([], fixture_resolver()) == []

    def test_unknown_division_drops_the_fixture(self):
        result = resolve_fixture_names(
            [fixture("Sp Lisbon", "Porto", division="ZZ")], fixture_resolver()
        )
        assert result == []

    def test_a_missing_resolver_drops_everything(self):
        """Fail closed: unverifiable names must never reach the model."""
        assert resolve_fixture_names([fixture("Sp Lisbon", "Porto")], None) == []


class TestFlashScoreIntegration:
    """The FlashScore fallback must run its fixtures through resolution."""

    def test_flashscore_fixtures_are_resolved(self, monkeypatch):
        import src.scrapers.fixtures_fetcher as module

        scraped = [fixture("Sporting CP", "Porto")]
        monkeypatch.setattr(
            module, "_scrape_flashscore_fixtures", lambda *a, **kw: scraped
        )
        monkeypatch.setattr(
            module, "_build_fixture_name_resolver", lambda: fixture_resolver(APPROVED)
        )
        result = module._fetch_flashscore_fixtures("01/05/2026", ["P1"])
        assert [f.home_team for f in result] == ["Sp Lisbon"]

    def test_unresolvable_flashscore_fixtures_are_dropped(self, monkeypatch):
        import src.scrapers.fixtures_fetcher as module

        monkeypatch.setattr(
            module,
            "_scrape_flashscore_fixtures",
            lambda *a, **kw: [fixture("Sporting CP", "Porto")],
        )
        monkeypatch.setattr(
            module, "_build_fixture_name_resolver", lambda: fixture_resolver()
        )
        assert module._fetch_flashscore_fixtures("01/05/2026", ["P1"]) == []

    def test_resolver_construction_failure_drops_everything(self, monkeypatch):
        import src.scrapers.fixtures_fetcher as module

        monkeypatch.setattr(
            module,
            "_scrape_flashscore_fixtures",
            lambda *a, **kw: [fixture("Sp Lisbon", "Porto")],
        )
        monkeypatch.setattr(module, "_build_fixture_name_resolver", lambda: None)
        assert module._fetch_flashscore_fixtures("01/05/2026", ["P1"]) == []


class TestSupabaseAvailability:
    """The pipeline runs in CI, where Supabase credentials may be absent."""

    def test_alias_repository_falls_back_to_the_seed_without_supabase(
        self, monkeypatch
    ):
        import src.scrapers.fixtures_fetcher as module
        from src.teams.resolver import ChainedTeamAliasRepository

        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        config = MagicMock()
        config.teams.aliases = CONFIG
        repository = module._build_alias_repository(config)
        assert isinstance(repository, ChainedTeamAliasRepository)
        # The reviewed seed still answers, so committed aliases keep working.
        assert repository.get_aliases() == []

    def test_unresolved_sink_is_absent_without_supabase(self, monkeypatch):
        import src.scrapers.fixtures_fetcher as module

        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        assert module._build_unresolved_sink() is None

    def test_resolution_still_works_without_a_review_queue(self, monkeypatch):
        """No queue is a lost report, never a lost verification."""
        import src.scrapers.fixtures_fetcher as module

        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        resolver = FixtureNameResolver(
            TeamNameResolver(
                canonical_teams=CANONICAL,
                alias_repository=FakeAliasRepository(APPROVED),
                config=CONFIG,
            ),
            sink=module._build_unresolved_sink(),
        )
        result = resolve_fixture_names([fixture("Sporting CP", "Porto")], resolver)
        assert [f.home_team for f in result] == ["Sp Lisbon"]


class TestResolverConstruction:

    def test_builds_a_resolver_from_config(self, monkeypatch):
        import src.scrapers.fixtures_fetcher as module

        config = MagicMock()
        config.teams.registry_path = "src/backend/data/teams_registry.json"
        config.teams.aliases = CONFIG
        monkeypatch.setattr(module, "_load_project_config", lambda: config)
        monkeypatch.setattr(module, "_build_alias_repository", lambda cfg: FakeAliasRepository())
        assert isinstance(module._build_fixture_name_resolver(), FixtureNameResolver)

    def test_returns_none_when_config_is_unavailable(self, monkeypatch):
        import src.scrapers.fixtures_fetcher as module

        def _boom():
            raise RuntimeError("no config")

        monkeypatch.setattr(module, "_load_project_config", _boom)
        assert module._build_fixture_name_resolver() is None

    def test_returns_none_when_the_registry_is_empty(self, monkeypatch):
        import src.scrapers.fixtures_fetcher as module

        config = MagicMock()
        config.teams.registry_path = "does/not/exist.json"
        config.teams.aliases = CONFIG
        monkeypatch.setattr(module, "_load_project_config", lambda: config)
        monkeypatch.setattr(module, "_build_alias_repository", lambda cfg: FakeAliasRepository())
        assert module._build_fixture_name_resolver() is None
