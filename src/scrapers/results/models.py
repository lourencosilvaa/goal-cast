"""The domain objects the results track passes around.

Frozen dataclasses rather than Pydantic models: these are the *internal*
vocabulary, produced by providers and consumed by the tracker. The HTTP
contract has its own Pydantic schemas in :mod:`src.results_service.api.results`
so that changing a wire field never silently changes what a provider must
produce, and vice versa.

Immutability is not decoration. A live snapshot is diffed against its
successor to derive events; if a match could be mutated in place, the previous
snapshot would change under the diff and produce goals that never happened.

Team names are carried **exactly as the provider spells them**, following the
same rule as European fixtures (``src/scrapers/european/providers.py``).
Resolution to canonical names happens above this layer, through the
refuse-to-guess resolver, never by a provider picking a spelling it likes.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MatchStatus(str, Enum):
    """Where a match is in its life, normalised across providers.

    A ``str`` enum so a value survives JSON serialisation unchanged and a
    comparison against the literal ``"live"`` keeps working.

    ``PAUSED`` covers half-time and any other in-match stoppage: the match is
    under way but the clock is not running, which is materially different from
    ``LIVE`` for anyone watching a score board.
    """

    SCHEDULED = "scheduled"
    LIVE = "live"
    PAUSED = "paused"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    """A change detected between two consecutive live snapshots."""

    GOAL = "goal"
    KICKOFF = "kickoff"
    FULL_TIME = "full_time"


@dataclass(frozen=True)
class MatchResult:
    """One match, in provider spelling, at one moment in time.

    ``home_goals``/``away_goals`` are ``None`` — not ``0`` — before kick-off.
    Zero is a scoreline; ``None`` is the absence of one, and collapsing the two
    would make a scheduled match indistinguishable from a goalless first
    minute.
    """

    league: str
    kickoff: datetime
    home_team: str
    away_team: str
    status: MatchStatus
    home_goals: int | None = None
    away_goals: int | None = None
    #: Provider's own clock reading ("67'", "HT"). Empty when it supplies
    #: none — the football-data.org free tier never does, and deriving a
    #: minute from kick-off would present arithmetic as observation.
    minute: str = ""
    #: Which provider answered, kept per match so a merged view stays honest.
    source: str = ""
    source_id: str = ""

    @property
    def key(self) -> tuple[str, str, str, str]:
        """Identity across snapshots.

        The kick-off **date** is part of it, not the timestamp: a provider may
        adjust the exact time of a match in progress, and two snapshots of the
        same match must still diff against each other. The date is coarse
        enough to survive that and fine enough to keep the two league meetings
        of the same pair apart.
        """
        return (
            self.league,
            self.kickoff.date().isoformat(),
            self.home_team,
            self.away_team,
        )


@dataclass(frozen=True)
class HistoryQuery:
    """One league-season of finished matches.

    ``season`` uses football-data's ``YYZZ`` form ("2526"), the same spelling
    as ``data.seasons``, so a season string never needs translating between
    this track and the training corpus.
    """

    league: str
    season: str


@dataclass(frozen=True)
class LiveSnapshot:
    """Everything one live query returned, and when."""

    fetched_at: datetime
    matches: tuple[MatchResult, ...]
    #: The provider that answered; empty when none did.
    source: str


@dataclass(frozen=True)
class MatchEvent:
    """Something that changed, with the match as it looked afterwards."""

    type: EventType
    match: MatchResult
    detected_at: datetime


@dataclass(frozen=True)
class LiveUpdate:
    """What the tracker serves: a snapshot, its events, and its honesty flag.

    ``stale`` says the snapshot is older than the configured window — every
    live provider has been failing — so the caller can show the last known
    scores while telling the user they may have moved on.
    """

    snapshot: LiveSnapshot
    events: tuple[MatchEvent, ...]
    stale: bool
