"""The JSON the results service speaks, as Pydantic models.

Deliberately separate from :mod:`src.scrapers.results.models`, which is the
service's internal vocabulary. Two reasons:

* The internal models are frozen dataclasses built for diffing snapshots. The
  wire needs validation and JSON serialisation, which is a different job.
* Keeping them apart means renaming an internal field cannot silently change
  the payload the frontend already renders, and adding a wire field cannot
  impose a requirement on every provider.

Both sides of the HTTP boundary import *this* module — the service to build
responses, the main app's gateway to parse them — so the contract is one
object rather than two that drift apart. ``tests/test_results_contract.py``
holds that line.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class MatchModel(BaseModel):
    """One match as the service reports it."""

    league: str
    kickoff: datetime
    home_team: str
    away_team: str
    #: One of ``scheduled``/``live``/``paused``/``finished``/``postponed``/
    #: ``cancelled``. A plain string rather than an enum so an unfamiliar
    #: status from a newer service version reaches the UI instead of failing
    #: the whole response.
    status: str
    #: ``null`` before kick-off — not ``0``, which is a scoreline.
    home_goals: int | None = None
    away_goals: int | None = None
    #: Provider's clock ("67'"), empty when it supplies none.
    minute: str = ""
    #: Minutes played, in ``[0, 90]``. Always present for a match in progress,
    #: because the in-play prediction needs it and the football-data.org free
    #: tier never sends one. ``minute_estimated`` says which it is, so a UI can
    #: show "~57'" rather than passing arithmetic off as the official clock.
    elapsed_minutes: int = 0
    minute_estimated: bool = False
    #: Which provider produced this match.
    source: str = ""
    source_id: str = ""


class EventModel(BaseModel):
    """A change detected between two consecutive live snapshots."""

    #: ``goal``, ``kickoff`` or ``full_time``.
    type: str
    match: MatchModel
    detected_at: datetime


class LiveResultsResponse(BaseModel):
    """``GET /live``.

    ``stale`` is part of the contract rather than a detail: when every live
    provider is failing, the service keeps serving the last snapshot, and the
    caller has to be able to tell that apart from a current one.
    """

    fetched_at: datetime
    #: The provider that answered; empty when none did.
    source: str = ""
    stale: bool = False
    matches: list[MatchModel] = Field(default_factory=list)
    events: list[EventModel] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    """``GET /history``."""

    league: str
    season: str
    source: str = ""
    matches: list[MatchModel] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """``GET /health`` — unauthenticated, and deliberately shallow.

    It reports that this process is up and which build it is running. It does
    not touch a provider: a health check that polled upstream would spend
    quota on every probe and take the service down whenever a third party
    was.
    """

    status: str
    commit: str
    build_time: str
