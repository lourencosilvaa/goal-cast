import pandas as pd  # noqa: F401  (kept for parity; not strictly required)

from config.config_loader import InternationalFlashScoreConfig
from src.scrapers.base_scraper import FlashScoreFixture
from src.scrapers.fixtures_fetcher import Fixture
from src.scrapers.international_fixtures_fetcher import InternationalFixturesFetcher


class _FakeScraper:
    """Deterministic in-memory FlashScore stand-in."""

    def __init__(self, fixtures_by_code: dict[str, list[FlashScoreFixture]]) -> None:
        self._fixtures = fixtures_by_code
        self.scraped_codes: list[str] = []

    def get_available_leagues(self) -> dict[str, str]:
        return {code: code for code in self._fixtures}

    def scrape_fixtures(self, league_code: str) -> list[FlashScoreFixture]:
        self.scraped_codes.append(league_code)
        return self._fixtures.get(league_code, [])


def _fs_fixture(home: str, away: str, dt: str) -> FlashScoreFixture:
    return FlashScoreFixture(
        match_id=f"{home}-{away}",
        home_team=home,
        away_team=away,
        league="World Cup",
        country="World",
        match_datetime=dt,
        status="scheduled",
        home_score=None,
        away_score=None,
        source_url="https://example.test",
    )


def _config() -> InternationalFlashScoreConfig:
    return InternationalFlashScoreConfig(
        leagues={"WC": "world/world-cup", "EURO": "europe/euro"}
    )


class TestInternationalFixturesFetcher:
    def test_returns_fixtures_without_odds(self):
        scraper = _FakeScraper(
            {"WC": [_fs_fixture("Portugal", "Spain", "2026-06-14T20:00:00")]}
        )
        fetcher = InternationalFixturesFetcher(_config(), scraper)
        fixtures = fetcher.fetch_upcoming(tournaments=["WC"])
        assert len(fixtures) == 1
        fx = fixtures[0]
        assert isinstance(fx, Fixture)
        assert fx.home_team == "Portugal"
        assert fx.has_odds is False
        assert fx.b365_home is None

    def test_defaults_to_all_configured_tournaments(self):
        scraper = _FakeScraper(
            {
                "WC": [_fs_fixture("Brazil", "Argentina", "2026-06-15T18:00:00")],
                "EURO": [_fs_fixture("France", "Italy", "2026-06-16T18:00:00")],
            }
        )
        fetcher = InternationalFixturesFetcher(_config(), scraper)
        fixtures = fetcher.fetch_upcoming()
        assert {"WC", "EURO"} == set(scraper.scraped_codes)
        assert len(fixtures) == 2

    def test_filters_by_target_date(self):
        scraper = _FakeScraper(
            {
                "WC": [
                    _fs_fixture("Brazil", "Argentina", "2026-06-15T18:00:00"),
                    _fs_fixture("Spain", "Germany", "2026-06-20T18:00:00"),
                ]
            }
        )
        fetcher = InternationalFixturesFetcher(_config(), scraper)
        fixtures = fetcher.fetch_upcoming(
            tournaments=["WC"], target_date="15/06/2026"
        )
        assert len(fixtures) == 1
        assert fixtures[0].away_team == "Argentina"

    def test_unknown_tournament_is_skipped(self):
        scraper = _FakeScraper({"WC": []})
        fetcher = InternationalFixturesFetcher(_config(), scraper)
        fixtures = fetcher.fetch_upcoming(tournaments=["DOES_NOT_EXIST"])
        assert fixtures == []

    def test_unavailable_source_is_skipped(self):
        from src.scrapers.flashscore.exceptions import FlashScoreUnavailableError

        class _RaisingScraper(_FakeScraper):
            def scrape_fixtures(self, league_code: str):
                raise FlashScoreUnavailableError("down")

        fetcher = InternationalFixturesFetcher(
            _config(), _RaisingScraper({"WC": []})
        )
        assert fetcher.fetch_upcoming(tournaments=["WC"]) == []

    def test_unparseable_datetime_with_target_is_excluded(self):
        scraper = _FakeScraper({"WC": [_fs_fixture("A", "B", "not-a-date")]})
        fetcher = InternationalFixturesFetcher(_config(), scraper)
        assert fetcher.fetch_upcoming(tournaments=["WC"], target_date="01/01/2026") == []

    def test_unparseable_datetime_without_target_is_kept(self):
        scraper = _FakeScraper({"WC": [_fs_fixture("A", "B", "TBD")]})
        fetcher = InternationalFixturesFetcher(_config(), scraper)
        fixtures = fetcher.fetch_upcoming(tournaments=["WC"])
        assert len(fixtures) == 1
        assert fixtures[0].date == "TBD"
        assert fixtures[0].time == ""


class TestFactory:
    def test_build_returns_wired_fetcher(self):
        from src.scrapers.international_fixtures_fetcher import (
            build_international_fixtures_fetcher,
        )

        fetcher = build_international_fixtures_fetcher()
        assert isinstance(fetcher, InternationalFixturesFetcher)
        assert len(fetcher.config.leagues) > 0
