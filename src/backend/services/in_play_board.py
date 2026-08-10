"""Joining the live board to the stored predictions, so matches can be re-priced.

:mod:`src.backend.services.in_play` prices one match given its pre-match
scoring rates and its current state. Neither of those two things lives in the
same place: the rates are in the prediction written offline into Supabase, the
state comes from the results service over HTTP. This module is what puts them
in the same object.

**The hard part is identity, not arithmetic.** Every prediction in this project
is keyed by the football-data.co.uk spelling of a club — "Sp Lisbon", "Ein
Frankfurt" — while the live board reports whatever its provider calls it. The
two coincide often enough to be tempting and differ often enough to be
dangerous, so names go through :mod:`src.teams.resolver`, which resolves an
exact match or a human-approved alias and *refuses to guess* at anything else.

**A match that cannot be priced is reported, not dropped.** Three matches live
and two in-play cards is indistinguishable from a bug unless the third says why
it is missing. Every skipped match therefore comes back in ``unpriced`` with a
reason, which is also what makes a missing alias visible to whoever can approve
it.

Nothing here is cached. The live board's freshness is owned by the results
service's own TTL and the predictions by ``PredictionService``; a third cache
in between would only make "how old is this?" unanswerable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Mapping, Sequence

from src.backend.services.in_play import (
    ExpectedGoals,
    InPlayCalculator,
    InPlayForecast,
    LiveState,
)
from src.contracts.results import LiveResultsResponse, MatchModel
from src.models.outcome_model import OutcomeProbabilities


class PredictionSource(ABC):
    """The stored predictions for one league, as written offline.

    A row rather than a model on purpose: the writer's payload evolves, and
    this reads two fields out of it. Narrowing the dependency to "give me the
    rows" keeps the board out of the prediction service's shape.
    """

    @abstractmethod
    def matches(self, league: str, match_date: str) -> Sequence[Mapping[str, Any]]:
        """Prediction rows for one league-day, or empty when there are none.

        ``match_date`` is ``DD/MM/YYYY`` — the form the writer files rows
        under, and the reason this takes a date at all: a match that kicked
        off at 21:00 is still being played after the server's clock has
        rolled over, and "today" would then look in the wrong day.
        """


class NameMatcher(ABC):
    """Maps a provider's spelling onto the canonical one, or admits defeat."""

    @abstractmethod
    def canonical(self, league: str, raw_name: str) -> str | None:
        """The canonical name, or ``None`` when it cannot be resolved safely."""


@dataclass(frozen=True)
class PricedMatch:
    """A live match with both numbers: what was expected, and what is expected now."""

    league: str
    #: Canonical spelling — the key the frontend joins its own rows on.
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    elapsed_minutes: int
    #: True when the minute was derived from kick-off rather than observed.
    minute_estimated: bool
    status: str
    pre_match: OutcomeProbabilities
    forecast: InPlayForecast


@dataclass(frozen=True)
class UnpricedMatch:
    """A live match that could not be re-priced, and why.

    Carries the *provider's* spelling rather than a canonical one: when the
    reason is an unresolved name, that spelling is the whole diagnosis.
    """

    league: str
    home_team: str
    away_team: str
    reason: str


@dataclass(frozen=True)
class InPlayBoardResult:
    """Everything currently in progress, priced where it could be."""

    fetched_at: datetime
    #: Carried from the live board: the results service serves its last
    #: snapshot when every provider is failing, and a re-priced stale score is
    #: still stale.
    stale: bool = False
    matches: list[PricedMatch] = field(default_factory=list)
    unpriced: list[UnpricedMatch] = field(default_factory=list)


class InPlayBoard:
    """Re-prices every match in progress that can be identified confidently."""

    #: A match is priceable while it is being played. ``paused`` is half-time,
    #: which is a match in progress with a score and 45 minutes left.
    LIVE_STATUSES: ClassVar[frozenset[str]] = frozenset({"live", "paused"})

    #: How the prediction writer files a day. Kept next to the reader that
    #: needs it rather than in configuration: it is the shape of data already
    #: in Supabase, not a setting anyone may change.
    DATE_FORMAT: ClassVar[str] = "%d/%m/%Y"

    #: Reasons a live match came back unpriced.
    UNKNOWN_TEAM: ClassVar[str] = "unknown_team"
    NO_PREDICTION: ClassVar[str] = "no_prediction"
    NO_EXPECTED_GOALS: ClassVar[str] = "no_expected_goals"

    def __init__(
        self,
        gateway: Any,
        predictions: PredictionSource,
        names: NameMatcher,
        calculator: InPlayCalculator | None = None,
    ) -> None:
        self._gateway = gateway
        self._predictions = predictions
        self._names = names
        self._calculator = calculator or InPlayCalculator()

    def build(self, leagues: Sequence[str]) -> InPlayBoardResult:
        """Price every in-progress match in ``leagues`` (empty means all).

        Gateway failures propagate. An unreachable results service and a night
        with no football produce the same empty board, and only one of them is
        something a user should be told.
        """
        board: LiveResultsResponse = self._gateway.live(list(leagues))
        result = InPlayBoardResult(fetched_at=board.fetched_at, stale=board.stale)
        #: Read once per league-day that is actually playing, not per served
        #: league: each miss is a Supabase round trip.
        rows: dict[tuple[str, str], dict[tuple[str, str], Mapping[str, Any]]] = {}

        for match in board.matches:
            if not self._is_in_progress(match):
                continue
            self._place(match, rows, result)
        return result

    # ── one match ────────────────────────────────────────────────────────

    def _place(
        self,
        match: MatchModel,
        rows: dict[tuple[str, str], dict[tuple[str, str], Mapping[str, Any]]],
        result: InPlayBoardResult,
    ) -> None:
        """Price ``match`` into ``result``, or record why it could not be."""
        home = self._names.canonical(match.league, match.home_team)
        away = self._names.canonical(match.league, match.away_team)
        if home is None or away is None:
            result.unpriced.append(self._unpriced(match, self.UNKNOWN_TEAM))
            return

        day = (match.league, match.kickoff.strftime(self.DATE_FORMAT))
        if day not in rows:
            rows[day] = self._index(self._predictions.matches(*day))
        row = rows[day].get((home, away))
        if row is None:
            result.unpriced.append(self._unpriced(match, self.NO_PREDICTION))
            return

        expected = self._expected_goals(row)
        if expected is None:
            result.unpriced.append(self._unpriced(match, self.NO_EXPECTED_GOALS))
            return

        result.matches.append(
            PricedMatch(
                league=match.league,
                home_team=home,
                away_team=away,
                home_goals=int(match.home_goals or 0),
                away_goals=int(match.away_goals or 0),
                elapsed_minutes=match.elapsed_minutes,
                minute_estimated=match.minute_estimated,
                status=match.status,
                pre_match=self._pre_match(row),
                forecast=self._calculator.forecast(
                    expected,
                    LiveState(
                        home_goals=int(match.home_goals or 0),
                        away_goals=int(match.away_goals or 0),
                        elapsed_minutes=match.elapsed_minutes,
                    ),
                ),
            )
        )

    def _is_in_progress(self, match: MatchModel) -> bool:
        """Playing, and with a score to price from.

        ``None`` goals are not 0-0. A provider that reports a match as live
        before it has published a scoreline would otherwise have a goalless
        draw invented for it.
        """
        return (
            match.status in self.LIVE_STATUSES
            and match.home_goals is not None
            and match.away_goals is not None
        )

    @staticmethod
    def _unpriced(match: MatchModel, reason: str) -> UnpricedMatch:
        return UnpricedMatch(
            league=match.league,
            home_team=match.home_team,
            away_team=match.away_team,
            reason=reason,
        )

    # ── the stored prediction ────────────────────────────────────────────

    @staticmethod
    def _index(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[str, str], Mapping[str, Any]]:
        """``(home, away)`` → row. Both names, because a league plays a team
        twice a season and only the pair identifies the fixture."""
        return {
            (str(row.get("home_team", "")), str(row.get("away_team", ""))): row
            for row in rows
        }

    @classmethod
    def _expected_goals(cls, row: Mapping[str, Any]) -> ExpectedGoals | None:
        """The Poisson rates, or ``None`` when the row carries none.

        There is no fallback. Substituting a league average would produce a
        confident live number for a fixture the model never priced, which is
        the failure mode this project treats as worse than a gap.
        """
        raw = row.get("expected_goals")
        if not isinstance(raw, Mapping):
            return None
        home = cls._number(raw.get("home"))
        away = cls._number(raw.get("away"))
        if home is None or away is None:
            return None
        return ExpectedGoals(home=home, away=away)

    @classmethod
    def _pre_match(cls, row: Mapping[str, Any]) -> OutcomeProbabilities:
        """What the model said before kick-off, shown next to the live number
        so the movement is legible."""
        raw = row.get("probabilities")
        probabilities = raw if isinstance(raw, Mapping) else {}
        return OutcomeProbabilities(
            home_win=cls._number(probabilities.get("home_win")) or 0.0,
            draw=cls._number(probabilities.get("draw")) or 0.0,
            away_win=cls._number(probabilities.get("away_win")) or 0.0,
        )

    @staticmethod
    def _number(value: Any) -> float | None:
        """A float, or ``None`` for anything that is not one.

        The row comes from JSON written by a separate process, so a string or
        a null is a real possibility and must not raise here.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)
