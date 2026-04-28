from src.scrapers.base_scraper import BaseFixtureScraper, FlashScoreFixture
from src.scrapers.flashscore.exceptions import (
    FlashScoreHttpError,
    FlashScoreUnavailableError,
)
from src.scrapers.flashscore.http_client import FlashScoreHttpClient
from src.scrapers.flashscore.parser import FlashScoreParser
from src.scrapers.flashscore.playwright_client import FlashScorePlaywrightClient


class FlashScoreScraper(BaseFixtureScraper):
    """Orchestrates FlashScore scraping: HTTP primary, Playwright fallback."""

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
        league_slug = self.get_available_leagues().get(league_code, "")
        league_name, _, country = league_slug.partition("/")
        league_label = (
            country.replace("-", " ").title()
            if country
            else league_name.replace("-", " ").title()
        )
        country_label = league_name.replace("-", " ").title()

        raw: str | None = None

        if self.config.http_enabled:  # type: ignore[attr-defined]
            try:
                fetch = (
                    self._http.fetch_fixtures
                    if method == "fixtures"
                    else self._http.fetch_results
                )
                raw = fetch(league_code)
            except FlashScoreHttpError:
                raw = None

        if raw is None and self.config.playwright_fallback_enabled:  # type: ignore[attr-defined]
            try:
                fetch_pw = (
                    self._playwright.fetch_fixtures
                    if method == "fixtures"
                    else self._playwright.fetch_results
                )
                raw = fetch_pw(league_code)
            except FlashScoreHttpError:
                raw = None

        if raw is None:
            raise FlashScoreUnavailableError(
                f"FlashScore unavailable for {league_code!r}: all strategies failed"
            )

        return self._parser.parse_fixtures(
            raw, league=league_label, country=country_label
        )
