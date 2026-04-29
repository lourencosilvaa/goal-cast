from datetime import datetime

from playwright.sync_api import sync_playwright

from src.scrapers.base_scraper import FlashScoreFixture
from src.scrapers.flashscore.exceptions import FlashScoreHttpError

_FIXTURES_PAGE = "football/{slug}/fixtures/"
_RESULTS_PAGE = "football/{slug}/results/"
_MATCH_ROW_SELECTOR = ".event__match"

# JavaScript run inside the browser to extract fixture rows from the rendered DOM.
# FlashScore delivers data via WebSocket in binary format; by the time the DOM is
# settled, all fixture rows are already painted by the framework.
_EXTRACT_SCRIPT = """() => {
    const nameAttr = '[data-testid="wcl-scores-simple-text-01"]';
    const rows = document.querySelectorAll('.event__match');
    return Array.from(rows).map(row => {
        const homeEl = row.querySelector('.event__homeParticipant ' + nameAttr);
        const awayEl = row.querySelector('.event__awayParticipant ' + nameAttr);
        const timeEl = row.querySelector('.event__time');
        const link   = row.querySelector('a[id]');
        const rowId  = link ? link.id : '';
        const matchId = rowId.replace('match-row-g_1_', '');
        const home = homeEl ? homeEl.innerText.trim() : '';
        const away = awayEl ? awayEl.innerText.trim() : '';
        const time = timeEl ? timeEl.innerText.trim() : '';
        return {matchId, home, away, time};
    }).filter(r => r.home && r.away);
}"""


class FlashScorePlaywrightClient:
    """Scrapes FlashScore fixture/result pages via Playwright and extracts
    FlashScoreFixture objects directly from the rendered DOM."""

    def __init__(self, config: object) -> None:
        self.config = config  # type: ignore[assignment]

    def scrape_fixtures(self, league_code: str) -> list[FlashScoreFixture]:
        return self._scrape(league_code, _FIXTURES_PAGE, status="scheduled")

    def scrape_results(self, league_code: str) -> list[FlashScoreFixture]:
        return self._scrape(league_code, _RESULTS_PAGE, status="finished")

    def _scrape(
        self, league_code: str, page_template: str, status: str
    ) -> list[FlashScoreFixture]:
        slug = self._resolve_slug(league_code)
        url = f"{self.config.base_url}/{page_template.format(slug=slug)}"  # type: ignore[attr-defined]
        league, country = self._derive_labels(slug)

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
                    "domcontentloaded",
                    timeout=self.config.request_timeout * 1000,  # type: ignore[attr-defined]
                )
                try:
                    page.wait_for_selector(
                        _MATCH_ROW_SELECTOR,
                        timeout=15000,
                    )
                except Exception:
                    pass  # no fixtures on this page — will return empty list

                raw_rows: list[dict] = page.evaluate(_EXTRACT_SCRIPT)
                browser.close()
        except FlashScoreHttpError:
            raise
        except Exception as exc:
            raise FlashScoreHttpError(
                f"Playwright scrape failed for {league_code}: {exc}"
            ) from exc

        return [
            self._build_fixture(row, league=league, country=country, status=status)
            for row in raw_rows
            if row.get("home") and row.get("away")
        ]

    def _build_fixture(
        self,
        row: dict,
        *,
        league: str,
        country: str,
        status: str,
    ) -> FlashScoreFixture:
        match_id = row.get("matchId", "")
        return FlashScoreFixture(
            match_id=match_id,
            home_team=row.get("home", ""),
            away_team=row.get("away", ""),
            league=league,
            country=country,
            match_datetime=self._parse_time(row.get("time", "")),
            status=status,
            home_score=None,
            away_score=None,
            source_url=(
                f"https://www.flashscore.com/match/{match_id}/" if match_id else ""
            ),
        )

    def _parse_time(self, raw: str) -> str:
        """Parse FlashScore time format 'DD.MM. HH:MM' into ISO datetime."""
        if not raw:
            return ""
        try:
            year = datetime.now().year
            dt = datetime.strptime(f"{raw.strip()} {year}", "%d.%m. %H:%M %Y")
            return dt.isoformat()
        except Exception:
            return raw

    def _derive_labels(self, slug: str) -> tuple[str, str]:
        """Derive (league_label, country_label) from a slug like 'portugal/liga-portugal'."""  # noqa: E501
        country_part, _, league_part = slug.partition("/")
        league_label = (
            league_part.replace("-", " ").title()
            if league_part
            else country_part.replace("-", " ").title()
        )
        country_label = country_part.replace("-", " ").title()
        return league_label, country_label

    def _resolve_slug(self, league_code: str) -> str:
        leagues: dict[str, str] = self.config.leagues  # type: ignore[attr-defined]
        if league_code not in leagues:
            raise FlashScoreHttpError(f"Unknown league code: {league_code!r}")
        return leagues[league_code]
