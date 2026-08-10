"""Estimating how far into a match we are, when the provider will not say.

football-data.org's free tier sends a ``score`` and a ``status`` and no clock
at all — verified against the live API on 2026-08-09. The Flashscore fallback
does send one, but it only answers for the leagues football-data does not
carry, so for most matches the minute has to be estimated from kick-off.

An estimate, and labelled as one. It cannot see stoppage time, a delayed start
or a floodlight failure, and the difference matters: this number is what an
in-play prediction divides the remaining goals by. So the rule is to under-
rather than over-state progress — claiming a match is later than it is throws
away probability mass that is still live.
"""

from datetime import datetime

from src.scrapers.results.elapsed import estimate_elapsed_minutes
from src.scrapers.results.models import MatchResult, MatchStatus

_KICKOFF = datetime(2026, 8, 9, 17, 0)


def _match(status=MatchStatus.LIVE, minute="") -> MatchResult:
    return MatchResult(
        league="P1",
        kickoff=_KICKOFF,
        home_team="Porto",
        away_team="Alverca",
        status=status,
        home_goals=1,
        away_goals=0,
        minute=minute,
        source="football-data.org",
    )


def _at(hour: int, minute: int) -> datetime:
    return datetime(2026, 8, 9, hour, minute)


class TestProviderSuppliedClock:
    def test_a_real_minute_is_preferred_over_any_estimate(self):
        """Flashscore sends the actual clock; never overrule an observation
        with arithmetic."""
        assert estimate_elapsed_minutes(_match(minute="67'"), _at(18, 30)) == 67

    def test_a_bare_number_is_accepted(self):
        assert estimate_elapsed_minutes(_match(minute="67"), _at(18, 30)) == 67

    def test_stoppage_time_is_read_as_its_base_minute(self):
        """"45+2" is the 45th minute plus stoppage; the remaining regulation
        time is what matters here, and it is the same as at 45."""
        assert estimate_elapsed_minutes(_match(minute="45+2"), _at(17, 50)) == 45

    def test_an_unparseable_clock_falls_back_to_the_estimate(self):
        assert estimate_elapsed_minutes(_match(minute="soon"), _at(17, 30)) == 30


class TestEstimatingFromKickoff:
    def test_the_first_half_is_wall_clock_time(self):
        assert estimate_elapsed_minutes(_match(), _at(17, 30)) == 30

    def test_half_time_is_subtracted_once_past_the_break(self):
        """An hour after kick-off, 15 of those minutes were the interval."""
        assert estimate_elapsed_minutes(_match(), _at(18, 0)) == 45

    def test_just_before_the_break_is_not_adjusted(self):
        assert estimate_elapsed_minutes(_match(), _at(17, 44)) == 44

    def test_a_match_cannot_run_past_full_time(self):
        """Stoppage is invisible here, and a minute past 90 would make the
        remaining time negative."""
        assert estimate_elapsed_minutes(_match(), _at(20, 0)) == 90

    def test_a_clock_before_kickoff_is_zero_not_negative(self):
        assert estimate_elapsed_minutes(_match(), _at(16, 30)) == 0


class TestStatus:
    def test_a_scheduled_match_has_not_started(self):
        """Even if the clock says kick-off passed — the provider says it has
        not, and the provider is watching."""
        match = _match(status=MatchStatus.SCHEDULED)
        assert estimate_elapsed_minutes(match, _at(17, 30)) == 0

    def test_a_paused_match_is_at_the_interval(self):
        match = _match(status=MatchStatus.PAUSED)
        assert estimate_elapsed_minutes(match, _at(17, 50)) == 45

    def test_a_finished_match_is_over(self):
        match = _match(status=MatchStatus.FINISHED)
        assert estimate_elapsed_minutes(match, _at(17, 30)) == 90

    def test_a_postponed_match_has_not_started(self):
        match = _match(status=MatchStatus.POSTPONED)
        assert estimate_elapsed_minutes(match, _at(19, 0)) == 0

    def test_a_cancelled_match_has_not_started(self):
        match = _match(status=MatchStatus.CANCELLED)
        assert estimate_elapsed_minutes(match, _at(19, 0)) == 0

    def test_a_finished_match_ignores_a_stale_provider_clock(self):
        """Some providers leave the last clock reading on a finished match."""
        match = _match(status=MatchStatus.FINISHED, minute="88'")
        assert estimate_elapsed_minutes(match, _at(19, 0)) == 90
