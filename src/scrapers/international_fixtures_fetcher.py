from datetime import datetime
from typing import TYPE_CHECKING, Optional

from config.config_loader import InternationalFlashScoreConfig
from src.scrapers.fixtures_fetcher import Fixture

if TYPE_CHECKING:
    from src.scrapers.base_scraper import BaseFixtureScraper


class InternationalFixturesFetcher:
    """Lists upcoming national-team fixtures via a FlashScore scraper.

    The scraper is injected (dependency inversion) so the fetcher is testable
    without network access and the underlying source stays replaceable. The
    Kaggle dataset carries no odds, so returned fixtures have ``None`` Bet365
    prices — surfaced as N/A downstream rather than a misleading ``0.0``.
    """

    def __init__(
        self,
        config: InternationalFlashScoreConfig,
        scraper: "BaseFixtureScraper",
    ) -> None:
        self.config = config
        self._scraper = scraper

    def _codes_to_scrape(self, tournaments: Optional[list[str]]) -> list[str]:
        available = set(self.config.leagues)
        if tournaments is None:
            return list(self.config.leagues)
        return [code for code in tournaments if code in available]

    @staticmethod
    def _matches_target_date(match_datetime: str, target_date: Optional[str]) -> bool:
        if target_date is None:
            return True
        try:
            fixture_date = datetime.fromisoformat(match_datetime).date()
            wanted = datetime.strptime(target_date, "%d/%m/%Y").date()
        except (ValueError, TypeError):
            return False
        return fixture_date == wanted

    def fetch_upcoming(
        self,
        tournaments: Optional[list[str]] = None,
        target_date: Optional[str] = None,
    ) -> list[Fixture]:
        """Return upcoming national-team fixtures (no odds).

        Args:
            tournaments: Tournament codes to include (keys of the configured
                slug map). ``None`` means every configured tournament.
            target_date: Optional ``DD/MM/YYYY`` filter; ``None`` keeps all.
        """
        from src.scrapers.flashscore.exceptions import FlashScoreUnavailableError

        fixtures: list[Fixture] = []
        for code in self._codes_to_scrape(tournaments):
            try:
                scraped = self._scraper.scrape_fixtures(code)
            except FlashScoreUnavailableError:
                continue

            for fs in scraped:
                if not self._matches_target_date(fs.match_datetime, target_date):
                    continue
                dt = fs.match_datetime
                fixtures.append(
                    Fixture(
                        division=code,
                        league=fs.league,
                        date=self._format_date(dt),
                        time=dt[11:16] if len(dt) >= 16 else "",
                        home_team=fs.home_team,
                        away_team=fs.away_team,
                        b365_home=None,
                        b365_draw=None,
                        b365_away=None,
                    )
                )
        return fixtures

    @staticmethod
    def _format_date(match_datetime: str) -> str:
        try:
            return datetime.fromisoformat(match_datetime).strftime("%d/%m/%Y")
        except (ValueError, TypeError):
            return match_datetime


def build_international_fixtures_fetcher() -> InternationalFixturesFetcher:
    """Construct a fetcher wired to a real FlashScore scraper from config."""
    from config.config_loader import load_config
    from src.scrapers.flashscore.playwright_client import FlashScorePlaywrightClient
    from src.scrapers.flashscore.scraper import FlashScoreScraper

    cfg = load_config()
    intl_fs = cfg.international.flashscore
    scrapers_cfg = cfg.scrapers
    flashscore_cfg = scrapers_cfg.flashscore

    class _MergedConfig:
        base_url = flashscore_cfg.base_url
        playwright_fallback_enabled = flashscore_cfg.playwright_fallback_enabled
        leagues = intl_fs.leagues
        user_agent = scrapers_cfg.user_agent
        request_timeout = scrapers_cfg.request_timeout

    merged = _MergedConfig()
    scraper = FlashScoreScraper(merged, FlashScorePlaywrightClient(merged))
    return InternationalFixturesFetcher(intl_fs, scraper)
