"""The optional Flashscore fallback.

Deliberately an adapter over the existing Playwright client rather than a new
scraper: the browser mechanics, the slug map and the timeout policy already
exist and are configured in one place. What this adds is the mapping from a
scraped row to a :class:`MatchResult`.

It is disabled in the shipped configuration. The Flashscore ToS prohibit
scraping and the Flashscore-only route was reverted in August 2026; the system
is fully functional without it, and these tests hold that line — nothing here
requires a browser, and the client is always injected.
"""

from datetime import date, datetime

from config.config_loader import ResultsProviderConfig
from src.scrapers.flashscore.exceptions import FlashScoreHttpError
from src.scrapers.flashscore.live_rows import LiveScoreRow
from src.scrapers.results.flashscore_live import FlashscoreLiveProvider
from src.scrapers.results.models import MatchStatus


class _Client:
    def __init__(self, rows=None, error=None):
        self._rows = rows or {}
        self._error = error
        self.requested: list[str] = []

    def scrape_live(self, league_code):
        self.requested.append(league_code)
        if self._error:
            raise self._error
        return list(self._rows.get(league_code, []))


def _row(**overrides) -> LiveScoreRow:
    defaults = dict(
        match_id="abc123",
        home_team="Porto",
        away_team="Alverca",
        home_goals=2,
        away_goals=0,
        minute="67'",
        kickoff="",
    )
    defaults.update(overrides)
    return LiveScoreRow(**defaults)


def _provider(client, enabled=True, today=None) -> FlashscoreLiveProvider:
    return FlashscoreLiveProvider(
        ResultsProviderConfig(enabled=enabled),
        client,
        today=today or (lambda: date(2026, 8, 9)),
    )


class TestMappingRows:
    def test_a_scored_row_becomes_a_live_match(self):
        provider = _provider(_Client({"P1": [_row()]}))
        match = provider.fetch_live(["P1"])[0]
        assert match.status is MatchStatus.LIVE

    def test_the_score_and_minute_survive_the_mapping(self):
        provider = _provider(_Client({"P1": [_row()]}))
        match = provider.fetch_live(["P1"])[0]
        assert (match.home_goals, match.away_goals, match.minute) == (2, 0, "67'")

    def test_the_league_code_is_carried_through(self):
        provider = _provider(_Client({"P1": [_row()]}))
        assert provider.fetch_live(["P1"])[0].league == "P1"

    def test_the_provider_attributes_its_own_answers(self):
        provider = _provider(_Client({"P1": [_row()]}))
        assert provider.fetch_live(["P1"])[0].source == "flashscore"

    def test_a_row_with_no_score_is_scheduled(self):
        provider = _provider(
            _Client({"P1": [_row(home_goals=None, away_goals=None, minute="")]})
        )
        assert provider.fetch_live(["P1"])[0].status is MatchStatus.SCHEDULED

    def test_a_finished_row_is_finished(self):
        provider = _provider(_Client({"P1": [_row(minute="Finished")]}))
        assert provider.fetch_live(["P1"])[0].status is MatchStatus.FINISHED

    def test_half_time_is_a_pause_not_a_stoppage(self):
        provider = _provider(_Client({"P1": [_row(minute="Half Time")]}))
        assert provider.fetch_live(["P1"])[0].status is MatchStatus.PAUSED

    def test_a_row_without_a_kickoff_falls_back_to_todays_date(self):
        """Flashscore's live board shows the minute, not the kick-off time. The
        date is today's by construction — that is what "live" means."""
        provider = _provider(_Client({"P1": [_row()]}))
        assert provider.fetch_live(["P1"])[0].kickoff.date() == date(2026, 8, 9)

    def test_a_row_with_a_kickoff_keeps_it(self):
        provider = _provider(_Client({"P1": [_row(kickoff="2026-08-09T17:00:00")]}))
        assert provider.fetch_live(["P1"])[0].kickoff == datetime(2026, 8, 9, 17, 0)

    def test_a_row_with_no_teams_is_dropped(self):
        provider = _provider(_Client({"P1": [_row(home_team="", away_team="")]}))
        assert provider.fetch_live(["P1"]) == []


class TestSeveralLeagues:
    def test_every_requested_league_is_scraped(self):
        client = _Client({"P1": [_row()], "E0": [_row(home_team="Arsenal")]})
        assert len(_provider(client).fetch_live(["P1", "E0"])) == 2

    def test_a_league_the_client_does_not_know_is_skipped(self):
        client = _Client(error=FlashScoreHttpError("Unknown league code: 'ZZ'"))
        assert _provider(client).fetch_live(["ZZ"]) == []

    def test_one_failing_league_does_not_lose_the_others(self):
        class _Partial(_Client):
            def scrape_live(self, league_code):
                if league_code == "E0":
                    raise FlashScoreHttpError("timeout")
                return super().scrape_live(league_code)

        client = _Partial({"P1": [_row()]})
        assert len(_provider(client).fetch_live(["P1", "E0"])) == 1


class TestKillSwitch:
    def test_a_disabled_provider_scrapes_nothing(self):
        client = _Client({"P1": [_row()]})
        assert _provider(client, enabled=False).fetch_live(["P1"]) == []
        assert client.requested == []

    def test_a_disabled_provider_reports_itself_as_disabled(self):
        assert _provider(_Client(), enabled=False).enabled is False

    def test_the_fallback_needs_no_api_key(self):
        """Unlike the API providers — it is a scraper, not an account."""
        assert _provider(_Client()).enabled is True
