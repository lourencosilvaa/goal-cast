from playwright.sync_api import sync_playwright

from src.scrapers.flashscore.exceptions import FlashScoreHttpError

_FIXTURES_PATH = "football/{slug}/#/fixtures"
_RESULTS_PATH = "football/{slug}/#/results"


class FlashScorePlaywrightClient:
    """Fetches FlashScore pages via Playwright headless browser (fallback strategy)."""

    def __init__(self, config: object) -> None:
        self.config = config  # type: ignore[assignment]

    def fetch_fixtures(self, league_code: str) -> str:
        return self._fetch(league_code, _FIXTURES_PATH)

    def fetch_results(self, league_code: str) -> str:
        return self._fetch(league_code, _RESULTS_PATH)

    def _fetch(self, league_code: str, path_template: str) -> str:
        slug = self._resolve_slug(league_code)
        url = f"{self.config.base_url}/{path_template.format(slug=slug)}"  # type: ignore[attr-defined]
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=self.config.user_agent,  # type: ignore[attr-defined]
                    viewport={"width": 1920, "height": 1080},
                )
                page = context.new_page()
                page.goto(url, timeout=self.config.request_timeout * 1000)  # type: ignore[attr-defined]
                page.wait_for_load_state(
                    "networkidle",
                    timeout=self.config.request_timeout * 1000,  # type: ignore[attr-defined]
                )
                html = page.content()
                browser.close()
                return html
        except Exception as exc:
            raise FlashScoreHttpError(
                f"Playwright fetch failed for {league_code}: {exc}"
            ) from exc

    def _resolve_slug(self, league_code: str) -> str:
        leagues: dict[str, str] = self.config.leagues  # type: ignore[attr-defined]
        if league_code not in leagues:
            raise FlashScoreHttpError(f"Unknown league code: {league_code!r}")
        return leagues[league_code]
