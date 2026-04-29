from src.scrapers.base_scraper import BaseFixtureScraper, FlashScoreFixture
from src.scrapers.flashscore.exceptions import (
    FlashScoreHttpError,
    FlashScoreUnavailableError,
)
from src.scrapers.flashscore.http_client import FlashScoreHttpClient
from src.scrapers.flashscore.parser import FlashScoreParser
from src.scrapers.flashscore.playwright_client import FlashScorePlaywrightClient


class FlashScoreScraper(BaseFixtureScraper):
    """Orchestrates FlashScore scraping.

    HTTP is attempted first (always fails — token auth is dead) so the
    Playwright client, which extracts fixtures directly from the rendered DOM,
    is the effective data source.
    """

    def __init__(
        self,
        config: object,
        http_client: FlashScoreHttpClient,
        playwright_client: FlashScorePlaywrightClient,
        parser: FlashScoreParser,
    ) -> None:
        self.config = config  # type: ignore[assignment]
        self._http = http_client
        self._playwright = playwright_client
        self._parser = parser

    @property
    def source_name(self) -> str:
        return "FlashScore"

    def get_available_leagues(self) -> dict[str, str]:
        return dict(self.config.leagues)  # type: ignore[attr-defined]

    def scrape_fixtures(self, league_code: str) -> list[FlashScoreFixture]:
        return self._scrape(league_code, method="fixtures")

    def scrape_results(self, league_code: str) -> list[FlashScoreFixture]:
        return self._scrape(league_code, method="results")

    def _scrape(self, league_code: str, method: str) -> list[FlashScoreFixture]:
        # HTTP path: always raises FlashScoreHttpError (token auth no longer works).
        # Kept so the http_enabled flag still has observable effect in tests.
        if self.config.http_enabled:  # type: ignore[attr-defined]
            try:
                fetch = (
                    self._http.fetch_fixtures
                    if method == "fixtures"
                    else self._http.fetch_results
                )
                fetch(league_code)
            except FlashScoreHttpError:
                pass

        # Playwright path: DOM extraction returns FlashScoreFixture objects directly.
        if self.config.playwright_fallback_enabled:  # type: ignore[attr-defined]
            try:
                scrape = (
                    self._playwright.scrape_fixtures
                    if method == "fixtures"
                    else self._playwright.scrape_results
                )
                return scrape(league_code)
            except FlashScoreHttpError:
                pass

        raise FlashScoreUnavailableError(
            f"FlashScore unavailable for {league_code!r}: all strategies failed"
        )
