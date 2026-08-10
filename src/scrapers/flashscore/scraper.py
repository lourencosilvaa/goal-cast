"""FlashScore fixture and result scraping.

One strategy, named honestly. This class used to take three collaborators and
present itself as an HTTP-then-browser fallback chain, but two of them did
nothing: the HTTP client raised on every call by its own design (FlashScore
moved the ``feed_sign`` token into a runtime JS object, so a static request can
never authenticate) and the exception was discarded a line later, while the
parser was constructed, injected, stored and never called at all.

What actually fetched anything was the Playwright client, reading the rendered
DOM. That is what remains. The wrapper still earns its place: it owns the
enabled flag, the league map and the translation of a scrape failure into
:class:`FlashScoreUnavailableError`, so callers do not need to know which
mechanism produced a page.
"""

from src.scrapers.base_scraper import BaseFixtureScraper, FlashScoreFixture
from src.scrapers.flashscore.exceptions import (
    FlashScoreHttpError,
    FlashScoreUnavailableError,
)
from src.scrapers.flashscore.playwright_client import FlashScorePlaywrightClient


class FlashScoreScraper(BaseFixtureScraper):
    """Reads FlashScore pages through a headless browser."""

    def __init__(
        self,
        config: object,
        playwright_client: FlashScorePlaywrightClient,
    ) -> None:
        self.config = config  # type: ignore[assignment]
        self._playwright = playwright_client

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
        if self.config.playwright_fallback_enabled:  # type: ignore[attr-defined]
            scrape = (
                self._playwright.scrape_fixtures
                if method == "fixtures"
                else self._playwright.scrape_results
            )
            try:
                return scrape(league_code)
            except FlashScoreHttpError:
                # Translated, not swallowed: the caller wants "FlashScore could
                # not answer", and anything that is not FlashScore's own
                # failure mode is a bug here and must surface as itself.
                pass

        raise FlashScoreUnavailableError(
            f"FlashScore unavailable for {league_code!r}: all strategies failed"
        )
