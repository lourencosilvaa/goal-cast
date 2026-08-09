"""Interfaces for discovering upcoming UEFA club-competition fixtures.

Fixtures and history are deliberately separate abstractions. History arrives in
bulk, rarely, and must never be fetched inside the training path
(:mod:`src.corpus.supplementary`). Fixtures arrive daily, expire, and come from
sources with hard quotas. Different cadence, different failure modes.

Team names are carried **exactly as the provider spells them**. Resolution to
canonical keys happens later, through the same refuse-to-guess resolver
everything else uses. Providers disagree — the same tie came back as
"FC Kairat" from The Odds API and "Kairat Almaty" from TheSportsDB — and
guessing which club is meant is how one team's history gets attributed to
another.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, ClassVar, Protocol


@dataclass(frozen=True)
class FixtureWindow:
    """The date range to look for fixtures in, inclusive at both ends."""

    start: date
    end: date

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment.date() <= self.end

    @property
    def start_iso(self) -> str:
        return self.start.isoformat()

    @property
    def end_iso(self) -> str:
        return self.end.isoformat()


@dataclass(frozen=True)
class EuropeanFixture:
    """One upcoming match, in provider spelling.

    ``source_id`` is the provider's own identifier where it has one. The Odds
    API needs it to price a specific event without paying for a second search,
    so it is kept rather than discarded as an implementation detail.
    """

    competition: str
    kickoff: datetime
    home_team: str
    away_team: str
    source: str
    source_id: str = ""


@dataclass
class QuotaReport:
    """What a provider's response headers said about the budget left.

    Every field is optional because every provider reports differently — and
    some report nothing at all. A provider that cannot say must not be forced
    to invent a number.
    """

    remaining: int | None = None
    used: int | None = None

    @staticmethod
    def _as_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_headers(
        cls, headers: Any, remaining_key: str, used_key: str = ""
    ) -> "QuotaReport":
        get = getattr(headers, "get", lambda _k, _d=None: None)
        return cls(
            remaining=cls._as_int(get(remaining_key)),
            used=cls._as_int(get(used_key)) if used_key else None,
        )


class Transport(Protocol):
    """The minimal HTTP surface a provider needs.

    Injected so that no test reaches the network and the retry/timeout policy
    lives in exactly one place.
    """

    def get(
        self, url: str, params: dict | None = ..., headers: dict | None = ...
    ) -> Any: ...


class FixtureProvider(ABC):
    """A source of upcoming fixtures for one or more competitions.

    Implementations are interchangeable and know nothing about each other; the
    chain composes them. A provider never raises for an expected condition —
    an out-of-season competition, a tier that does not cover a competition, a
    transient network error — because the chain's job is to move on, and an
    exception per empty competition would make that unreadable.
    """

    #: Human-facing name, used in logs and to attribute an answer.
    name: ClassVar[str] = "provider"

    def __init__(self, config: Any, transport: Transport, api_key: str) -> None:
        self._config = config
        self._transport = transport
        self._api_key = api_key
        self.quota = QuotaReport()

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._config, "enabled", True)) and bool(self._api_key)

    def fetch(
        self, window: FixtureWindow, competitions: list[str]
    ) -> list[EuropeanFixture]:
        """Fixtures for ``competitions`` inside ``window``, provider spelling."""
        if not self.enabled:
            return []
        results: list[EuropeanFixture] = []
        for competition in competitions:
            key = self._config.competitions.get(competition)
            if not key:
                # Not every provider carries every competition. Silence here is
                # correct: it is configuration, not a runtime failure.
                continue
            results.extend(self._fetch_one(competition, key, window))
        return results

    @abstractmethod
    def _fetch_one(
        self, competition: str, key: str, window: FixtureWindow
    ) -> list[EuropeanFixture]:
        """Fetch and parse a single competition."""

    def _report_failure(self, competition: str, reason: str) -> None:
        print(f"[{self.name}] {competition}: {reason}")


@dataclass
class ProviderFailure:
    """Why a provider produced nothing, for the chain to report."""

    provider: str
    reason: str
    details: dict = field(default_factory=dict)
