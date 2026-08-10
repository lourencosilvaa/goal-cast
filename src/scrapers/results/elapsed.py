"""How far into a match we are — observed if possible, estimated if not.

The in-play prediction divides the remaining expected goals by whatever this
returns, so it is the single number that decides how much of a match is still
to come. It deserves to be explicit about how confident it is.

**Observation beats arithmetic.** When a provider sends a clock, that is the
answer. The Flashscore fallback does; football-data.org's free tier does not —
verified against the live API on 2026-08-09, which returned a ``LIVE`` match
with a score and no clock field at all. Since the live chain returns the first
non-empty answer and football-data goes first, the practical split is: a real
clock for the leagues the free tier does not carry (Scotland, Belgium, Turkey,
Greece, where Flashscore answers), and an estimate for the ones it does.

**The estimate is deliberately conservative.** It cannot see stoppage time, a
delayed kick-off or a long VAR check, and each of those makes the true minute
*earlier* than wall-clock arithmetic suggests. Overstating progress throws away
probability mass that is still live — a 1-0 read as the 88th minute when it is
really the 80th understates the chance of an equaliser. So the half-time break
is subtracted, and the result is clamped to regulation time.

``status`` overrules both. A provider saying SCHEDULED while the clock says
kick-off has passed is a provider watching the match, and this is not.
"""

import re
from datetime import datetime, timezone
from typing import Final

from src.scrapers.results.models import MatchResult, MatchStatus

#: Regulation time. Stoppage is invisible to every source used here, so a
#: match is never reported as further along than this.
FULL_TIME_MINUTES: Final[int] = 90
#: Length of a half, and the point the interval begins.
HALF_TIME_MINUTES: Final[int] = 45
#: The interval, subtracted once wall-clock time has passed through it.
BREAK_MINUTES: Final[int] = 15
#: Leading digits of a clock reading: "67'", "67", "45+2" → 67, 67, 45.
_CLOCK: Final[re.Pattern[str]] = re.compile(r"^\s*(\d{1,3})")

#: Statuses whose elapsed time is a fact, not a calculation.
_FIXED: Final[dict[MatchStatus, int]] = {
    MatchStatus.SCHEDULED: 0,
    MatchStatus.POSTPONED: 0,
    MatchStatus.CANCELLED: 0,
    MatchStatus.PAUSED: HALF_TIME_MINUTES,
    MatchStatus.FINISHED: FULL_TIME_MINUTES,
}


def utc_now() -> datetime:
    """The current UTC instant, naive — the spelling kick-offs use.

    Providers send UTC timestamps that this project stores without a timezone
    (see ``FootballDataBase._parse_time``), so comparing against an aware
    ``now`` would raise. ``datetime.utcnow()`` would give the right value and
    is deprecated, hence the explicit conversion in one place.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def estimate_elapsed_minutes(match: MatchResult, now: datetime) -> int:
    """Minutes played, in ``[0, 90]``.

    ``now`` is injected rather than read from the clock so this is testable as
    policy, and so a caller scoring a whole board uses one consistent instant.
    """
    fixed = _FIXED.get(match.status)
    if fixed is not None:
        return fixed

    reported = _parse_clock(match.minute)
    if reported is not None:
        return _clamp(reported)

    return _clamp(_from_kickoff(match.kickoff, now))


def _from_kickoff(kickoff: datetime, now: datetime) -> int:
    """Wall-clock minutes since kick-off, less the interval if it has passed."""
    elapsed = int((now - kickoff).total_seconds() // 60)
    if elapsed > HALF_TIME_MINUTES + BREAK_MINUTES:
        return elapsed - BREAK_MINUTES
    # Inside the break itself, wall-clock time would run past 45' while the
    # match clock is stopped. Holding at 45 is the honest reading.
    if elapsed > HALF_TIME_MINUTES:
        return HALF_TIME_MINUTES
    return elapsed


def _parse_clock(raw: str) -> int | None:
    match = _CLOCK.match(raw or "")
    return int(match.group(1)) if match else None


def _clamp(minutes: int) -> int:
    return max(0, min(FULL_TIME_MINUTES, minutes))
