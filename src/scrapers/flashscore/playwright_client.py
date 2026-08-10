from datetime import datetime

from playwright.sync_api import sync_playwright

from src.scrapers.base_scraper import FlashScoreFixture
from src.scrapers.flashscore.exceptions import FlashScoreHttpError
from src.scrapers.flashscore.live_rows import LiveScoreRow

_FIXTURES_PAGE = "football/{slug}/fixtures/"
_RESULTS_PAGE = "football/{slug}/results/"
#: The competition's landing page is its live score board.
_LIVE_PAGE = "football/{slug}/"
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


# Live rows carry two things the fixture script ignores: the score cells and
# the stage cell (the running minute, "Half Time", "Finished", or — before
# kick-off — the scheduled time). Kept as a separate script rather than
# extending the fixture one, so the results track cannot change what the
# fixture pipeline extracts.
_LIVE_EXTRACT_SCRIPT = """() => {
    const nameAttr = '[data-testid="wcl-scores-simple-text-01"]';
    const rows = document.querySelectorAll('.event__match');
    return Array.from(rows).map(row => {
        const homeEl  = row.querySelector('.event__homeParticipant ' + nameAttr);
        const awayEl  = row.querySelector('.event__awayParticipant ' + nameAttr);
        const homeSc  = row.querySelector('.event__score--home');
        const awaySc  = row.querySelector('.event__score--away');
        const stageEl = row.querySelector('.event__stage--block')
                     || row.querySelector('.event__time');
        const link    = row.querySelector('a[id]');
        const rowId   = link ? link.id : '';
        return {
            matchId: rowId.replace('match-row-g_1_', ''),
            home: homeEl ? homeEl.innerText.trim() : '',
            away: awayEl ? awayEl.innerText.trim() : '',
            homeScore: homeSc ? homeSc.innerText.trim() : '',
            awayScore: awaySc ? awaySc.innerText.trim() : '',
            stage: stageEl ? stageEl.innerText.trim() : '',
        };
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

    def scrape_live(self, league_code: str) -> list[LiveScoreRow]:
        """Today's score board for one competition, as painted.

        Returns transport-shaped rows rather than fixtures: what "Half Time"
        or an empty score means is a decision for
        :class:`~src.scrapers.results.flashscore_live.FlashscoreLiveProvider`,
        not for the client that read the page.
        """
        raw_rows = self._extract(league_code, _LIVE_PAGE, _LIVE_EXTRACT_SCRIPT)
        return [self._build_live_row(row) for row in raw_rows if self._named(row)]

    def _scrape(
        self, league_code: str, page_template: str, status: str
    ) -> list[FlashScoreFixture]:
        slug = self._resolve_slug(league_code)
        league, country = self._derive_labels(slug)
        raw_rows = self._extract(league_code, page_template, _EXTRACT_SCRIPT)
        return [
            self._build_fixture(row, league=league, country=country, status=status)
            for row in raw_rows
            if self._named(row)
        ]

    def _extract(self, league_code: str, page_template: str, script: str) -> list[dict]:
        """Open one rendered page and run ``script`` inside it.

        Shared by fixtures, results and the live board so the browser
        lifecycle, the timeout policy and the error translation exist once.
        """
        slug = self._resolve_slug(league_code)
        url = f"{self.config.base_url}/{page_template.format(slug=slug)}"  # type: ignore[attr-defined]

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

                raw_rows: list[dict] = page.evaluate(script)
                browser.close()
        except FlashScoreHttpError:
            raise
        except Exception as exc:
            raise FlashScoreHttpError(
                f"Playwright scrape failed for {league_code}: {exc}"
            ) from exc

        return raw_rows

    @staticmethod
    def _named(row: dict) -> bool:
        return bool(row.get("home") and row.get("away"))

    def _build_live_row(self, row: dict) -> LiveScoreRow:
        return LiveScoreRow(
            match_id=row.get("matchId", ""),
            home_team=row.get("home", ""),
            away_team=row.get("away", ""),
            home_goals=self._parse_score(row.get("homeScore", "")),
            away_goals=self._parse_score(row.get("awayScore", "")),
            minute=row.get("stage", ""),
        )

    @staticmethod
    def _parse_score(raw: str) -> int | None:
        """A score, or ``None`` when the cell holds anything else.

        Before kick-off the cell is empty, and a postponed match shows "-".
        Neither is nil-nil, and reading them as zero would put a scoreline on
        a match that has not been played.
        """
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            return None

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
