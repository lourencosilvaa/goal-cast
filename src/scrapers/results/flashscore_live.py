"""The optional Flashscore fallback for live results.

**An adapter, not a scraper.** The browser mechanics, the slug map, the user
agent and the timeout all already exist in
:class:`~src.scrapers.flashscore.playwright_client.FlashScorePlaywrightClient`
and are configured in one place. What this file adds is the only part that is
about *results*: deciding that "Half Time" is a pause, that a row with no score
has not kicked off, and that a scraped row becomes a
:class:`~src.scrapers.results.models.MatchResult`.

**Off by default, and honest about why.** ``results.providers.flashscore``
ships as ``enabled: false``. The Flashscore ToS prohibit scraping, and the
Flashscore-only route for European fixtures was reverted in August 2026 for
that reason. It stays in the chain as a switchable last resort for the leagues
football-data.org's free tier does not carry at all — Scotland, Belgium,
Turkey, Greece — and nothing above it assumes it will answer.

No detection-evasion of any kind lives here: no proxy rotation, no fingerprint
spoofing, no CAPTCHA solving. That is a posture, recorded in the plan, not a
gap waiting to be filled.
"""

from datetime import date, datetime, time
from typing import Any, Callable, ClassVar

from src.scrapers.results.base import LiveResultsProvider
from src.scrapers.results.models import MatchResult, MatchStatus


class FlashscoreLiveProvider(LiveResultsProvider):
    """Maps rendered score-board rows to matches, one league at a time."""

    name: ClassVar[str] = "flashscore"

    #: Stage text → status. Matched case-insensitively on a prefix, because
    #: the board writes "Half Time", "Finished", "After Penalties" and so on,
    #: and the tail of the phrase carries no extra meaning for a score board.
    STAGE_PREFIXES: ClassVar[tuple[tuple[str, MatchStatus], ...]] = (
        ("half time", MatchStatus.PAUSED),
        ("ht", MatchStatus.PAUSED),
        ("break", MatchStatus.PAUSED),
        ("finished", MatchStatus.FINISHED),
        ("after et", MatchStatus.FINISHED),
        ("after pen", MatchStatus.FINISHED),
        ("ft", MatchStatus.FINISHED),
        ("postponed", MatchStatus.POSTPONED),
        ("cancelled", MatchStatus.CANCELLED),
        ("abandoned", MatchStatus.CANCELLED),
    )

    def __init__(
        self,
        config: Any,
        client: Any,
        today: Callable[[], date] | None = None,
    ) -> None:
        # No API key: this is a scraper, not an account, so `enabled` is the
        # configuration flag alone.
        super().__init__(config, transport=_NO_TRANSPORT, api_key="")
        self._client = client
        self._today = today or date.today

    def fetch_live(self, leagues: list[str]) -> list[MatchResult]:
        if not self.enabled:
            return []
        results: list[MatchResult] = []
        for league in leagues:
            results.extend(self._one_league(league))
        return results

    def _one_league(self, league: str) -> list[MatchResult]:
        try:
            rows = list(self._client.scrape_live(league))
        except Exception as exc:
            # One league failing must not lose the ones that worked — an
            # unknown slug and a browser timeout are both local to a league.
            self._report_failure(league, f"scrape failed ({exc})")
            return []
        return [
            match for row in rows if (match := self._to_match(league, row)) is not None
        ]

    def _to_match(self, league: str, row: Any) -> MatchResult | None:
        home = str(getattr(row, "home_team", "") or "").strip()
        away = str(getattr(row, "away_team", "") or "").strip()
        if not home or not away:
            return None
        minute = str(getattr(row, "minute", "") or "").strip()
        home_goals = getattr(row, "home_goals", None)
        away_goals = getattr(row, "away_goals", None)
        status = self._status(minute, home_goals, away_goals)
        return MatchResult(
            league=league,
            kickoff=self._kickoff(getattr(row, "kickoff", "")),
            home_team=home,
            away_team=away,
            status=status,
            home_goals=home_goals,
            away_goals=away_goals,
            # Only meaningful while the clock is running; a stage word like
            # "Finished" is a status, and repeating it as a minute would make
            # the UI print it twice.
            minute=minute if status is MatchStatus.LIVE else "",
            source=self.name,
        )

    @classmethod
    def _status(
        cls, minute: str, home_goals: int | None, away_goals: int | None
    ) -> MatchStatus:
        lowered = minute.lower()
        for prefix, status in cls.STAGE_PREFIXES:
            if lowered.startswith(prefix):
                return status
        if home_goals is None or away_goals is None:
            # No score on the board is how a not-yet-started match looks; the
            # cell holds the kick-off time instead.
            return MatchStatus.SCHEDULED
        return MatchStatus.LIVE

    def _kickoff(self, raw: Any) -> datetime:
        """The row's kick-off, or today at midnight.

        A live board carries the clock, not the kick-off, and by construction
        everything on it is today — that is what "live" means. Midnight is
        visibly a placeholder rather than an invented kick-off time.
        """
        text = str(raw or "").strip()
        if text:
            try:
                return datetime.fromisoformat(text).replace(tzinfo=None)
            except ValueError:
                pass
        return datetime.combine(self._today(), time.min)


class _NoTransport:
    """Refuses the HTTP surface this provider must never use.

    The base class takes a transport because every other source needs one.
    This one reaches the network through the Playwright client instead, and a
    silent ``None`` would only surface as an AttributeError somewhere less
    obvious than here.
    """

    def get(
        self, url: str, params: dict | None = None, headers: dict | None = None
    ) -> Any:
        raise NotImplementedError(
            "FlashscoreLiveProvider fetches through the Playwright client, "
            "not the HTTP transport"
        )


_NO_TRANSPORT = _NoTransport()
