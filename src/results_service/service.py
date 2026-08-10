"""What the endpoints call: validate the request, then ask the right chain.

The validation is the substance. Both endpoints take league codes from a query
string, and an unrecognised code must produce an explicit refusal rather than
an empty list — the two are indistinguishable to a caller, and a misspelled
code would otherwise return "no matches" forever, correctly and uselessly.

Everything is injected. The service does not know which providers exist, where
their keys came from, or whether a snapshot is persisted; that is
:mod:`src.results_service.factory`'s job. What is left here is policy.
"""

from dataclasses import dataclass
from typing import Sequence

from src.scrapers.results.live_tracker import LiveResultsTracker
from src.scrapers.results.models import HistoryQuery, LiveUpdate, MatchResult


class UnknownLeagueError(ValueError):
    """A league code that is not offered. Answered as 422, never as empty."""


class UnknownSeasonError(ValueError):
    """A season that is not configured. Answered as 422, never as empty."""


@dataclass(frozen=True)
class ResultsCatalogue:
    """What may be asked for.

    A value object rather than two constructor arguments, because they travel
    together everywhere and a service taking ``(history, tracker, leagues,
    seasons)`` is the parameter list §4.3 exists to prevent.
    """

    leagues: tuple[str, ...]
    seasons: tuple[str, ...]


@dataclass(frozen=True)
class HistoryResult:
    """One league-season of finished matches, with its provenance."""

    league: str
    season: str
    #: Which provider answered; empty when none did.
    source: str
    matches: tuple[MatchResult, ...]


class ResultsService:
    """History and live results for the codes this deployment offers."""

    def __init__(
        self,
        history: object,
        tracker: LiveResultsTracker,
        catalogue: ResultsCatalogue,
    ) -> None:
        self._history = history
        self._tracker = tracker
        self._catalogue = catalogue

    @property
    def catalogue(self) -> ResultsCatalogue:
        return self._catalogue

    @property
    def history_chain(self) -> object:
        """The assembled history chain — inspected by the wiring tests, which
        are the only place the configured order is checked end to end."""
        return self._history

    @property
    def live_chain(self) -> object:
        return self._tracker.provider

    def history(self, query: HistoryQuery) -> HistoryResult:
        self._check_leagues([query.league])
        if query.season not in self._catalogue.seasons:
            raise UnknownSeasonError(
                f"unknown season {query.season!r}. Configured seasons: "
                f"{', '.join(self._catalogue.seasons)}"
            )
        matches = self._history.fetch_history(query)  # type: ignore[attr-defined]
        return HistoryResult(
            league=query.league,
            season=query.season,
            source=str(getattr(self._history, "last_source", "") or ""),
            matches=tuple(matches),
        )

    def live(self, leagues: Sequence[str]) -> LiveUpdate:
        """Today's matches for ``leagues``, or for everything when empty.

        An empty request means "the whole board", which is what a score-board
        page wants and what the provider returns in a single call anyway.
        """
        wanted = self._deduplicate(leagues) or list(self._catalogue.leagues)
        self._check_leagues(wanted)
        return self._tracker.get_snapshot(wanted)

    # ── validation ───────────────────────────────────────────────────────

    def _check_leagues(self, leagues: Sequence[str]) -> None:
        unknown = [code for code in leagues if code not in self._catalogue.leagues]
        if unknown:
            raise UnknownLeagueError(
                f"unknown league(s) {', '.join(repr(c) for c in unknown)}. "
                f"Available: {', '.join(self._catalogue.leagues)}"
            )

    @staticmethod
    def _deduplicate(leagues: Sequence[str]) -> list[str]:
        """Order-preserving, so a repeated code costs one lookup, not two."""
        return list(dict.fromkeys(leagues))
