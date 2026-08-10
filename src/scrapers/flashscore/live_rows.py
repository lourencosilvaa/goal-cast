"""What a live score board row carries, before it means anything.

Kept apart from :class:`~src.scrapers.base_scraper.FlashScoreFixture` rather
than folded into it. That dataclass is the fixture pipeline's contract, its
``to_dict`` is written into stored JSON, and a live row needs a field it does
not have (the clock) while leaving several of its fields meaningless. Growing
a shared type to serve a second, differently-shaped job is how one pipeline's
change starts breaking the other.

This is transport-shaped, not domain-shaped: strings as the page spells them,
including the minute. Turning it into a
:class:`~src.scrapers.results.models.MatchResult` — deciding what "Half Time"
means — is the results provider's job.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveScoreRow:
    """One row of a rendered Flashscore score board."""

    match_id: str
    home_team: str
    away_team: str
    #: ``None`` before kick-off: the board shows a clock time, not a score.
    home_goals: int | None
    away_goals: int | None
    #: The stage cell as painted — "67'", "Half Time", "Finished", or the
    #: kick-off time for a match that has not started.
    minute: str
    #: ISO kick-off if the row exposed one; empty otherwise.
    kickoff: str = ""
