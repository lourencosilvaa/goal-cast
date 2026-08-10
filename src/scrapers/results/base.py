"""Interfaces for the two kinds of results source.

**Two ABCs, not one.** History and live are deliberately separate abstractions,
for the same reason ``src/scrapers/european/providers.py`` keeps history and
fixtures apart: different cadence and different failure modes. History is
answered from disk, is stable once written, and a wrong answer is a permanent
wrong answer. Live is answered from a rate-limited API, is worthless a minute
later, and its normal failure mode — "nothing is being played" — is
indistinguishable from an outage unless the layer above is told which provider
spoke. A single interface would force one of them to pretend.

The inherited contract from the fixtures chain applies unchanged: **a provider
never raises for an expected condition**. A league it does not carry, a season
out of range, a plan restriction, a transient network error — all return an
empty list and report. The chain's job is to move on to the next source, and
an exception per empty league would make that unreadable.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from src.scrapers.european.providers import QuotaReport, Transport
from src.scrapers.results.models import HistoryQuery, MatchResult

__all__ = [
    "HistoryProvider",
    "LiveResultsProvider",
    "QuotaReport",
    "ResultsProvider",
    "Transport",
]


class ResultsProvider(ABC):
    """What every results source has in common.

    The transport is injected — never constructed here — so no test reaches
    the network and the timeout/retry policy stays in the one place that owns
    it (:class:`src.scrapers.european.http.HttpTransport`).
    """

    #: Human-facing name. Used to attribute an answer and in failure reports.
    name: ClassVar[str] = "results-provider"

    def __init__(self, config: Any, transport: Transport, api_key: str = "") -> None:
        self._config = config
        self._transport = transport
        self._api_key = api_key
        #: What the last response's headers said about the remaining budget.
        self.quota = QuotaReport()

    @property
    def enabled(self) -> bool:
        """Whether this source should be consulted at all.

        Only the configuration flag here. Sources that additionally need a
        credential narrow this — a missing key means "not set up", which is a
        normal development state, not a failure to report on every call.
        """
        return bool(getattr(self._config, "enabled", True))

    def _report_failure(self, subject: str, reason: str) -> None:
        print(f"[results:{self.name}] {subject}: {reason}")


class HistoryProvider(ResultsProvider):
    """A source of finished matches for one league-season."""

    name: ClassVar[str] = "history-provider"

    @abstractmethod
    def fetch_history(self, query: HistoryQuery) -> list[MatchResult]:
        """Finished matches for ``query``, oldest first, provider spelling."""


class LiveResultsProvider(ResultsProvider):
    """A source of today's matches, in whatever state they are in."""

    name: ClassVar[str] = "live-provider"

    @abstractmethod
    def fetch_live(self, leagues: list[str]) -> list[MatchResult]:
        """Today's matches for ``leagues``, scheduled ones included.

        Scheduled matches are part of the answer on purpose: a score board
        that only lists matches already under way cannot show what is about to
        start, and the same request returns them anyway.
        """
