"""Results from football-data.org v4 — history and live.

Call shapes verified against the live API on 2026-08-09 rather than taken from
the documentation, and two of the findings shaped the code:

* **The live endpoint takes one request for the whole day.**
  ``GET /v4/matches?date=YYYY-MM-DD`` returned 15 matches spanning every
  competition in the plan (``resultSet.competitions == "BSA,DED,PPL"``). Asking
  per competition instead would multiply a single poll by the number of served
  leagues and would not fit the free tier's 10 requests per minute — which is
  the whole reason 60-second polling is viable.
* **The free tier sends no ``minute``, and says ``LIVE`` where the docs say
  ``IN_PLAY``.** A match in progress came back with a ``score`` and
  ``status: "LIVE"`` and no clock field at all. Both spellings are therefore
  mapped, and ``minute`` stays empty rather than being computed from kick-off:
  arithmetic presented as observation is worse than an absent field.

The auth header, the quota header and the *uncovered* memory are the same as
the fixtures provider (``src/scrapers/european/football_data.py``) because they
are the same API: 403 and 404 mean "your plan does not include this", which is
a fact about the plan, and repeating the call burns quota to be told again.
"""

from datetime import date, datetime
from typing import Any, Callable, ClassVar

from src.scrapers.results.base import (
    HistoryProvider,
    LiveResultsProvider,
    QuotaReport,
    Transport,
)
from src.scrapers.results.models import HistoryQuery, MatchResult, MatchStatus


class FootballDataBase:
    """Shared call shape, parsing and status mapping for the two providers.

    A mixin rather than a common superclass: history and live inherit from
    *different* ABCs on purpose (see :mod:`src.scrapers.results.base`), and
    folding them into one hierarchy just to share a JSON parser would undo
    that separation.
    """

    #: Attributed name. Both providers report as the same upstream service.
    name: ClassVar[str] = "football-data.org"

    AUTH_HEADER: ClassVar[str] = "X-Auth-Token"
    REMAINING_HEADER: ClassVar[str] = "x-requests-available-minute"
    OK: ClassVar[int] = 200
    #: Responses meaning "your plan does not include this", not "it broke".
    NOT_COVERED: ClassVar[tuple[int, ...]] = (403, 404)

    #: Upstream status → ours. ``LIVE`` and ``IN_PLAY`` are both sent in
    #: practice; ``AWARDED`` is a result awarded off the pitch, which is
    #: finished as far as a score board is concerned; ``SUSPENDED`` is a match
    #: stopped mid-play, which is a pause rather than a cancellation.
    STATUS_MAP: ClassVar[dict[str, MatchStatus]] = {
        "SCHEDULED": MatchStatus.SCHEDULED,
        "TIMED": MatchStatus.SCHEDULED,
        "IN_PLAY": MatchStatus.LIVE,
        "LIVE": MatchStatus.LIVE,
        "PAUSED": MatchStatus.PAUSED,
        "SUSPENDED": MatchStatus.PAUSED,
        "FINISHED": MatchStatus.FINISHED,
        "AWARDED": MatchStatus.FINISHED,
        "POSTPONED": MatchStatus.POSTPONED,
        "CANCELLED": MatchStatus.CANCELLED,
        "CANCELED": MatchStatus.CANCELLED,
    }

    def _get(self, path: str, params: dict) -> Any:
        """One GET, or ``None`` when the answer is unusable.

        Returns the raw response so callers can distinguish a plan restriction
        from a server error; ``None`` means the request never produced one.
        """
        url = f"{self._config.base_url}{path}"  # type: ignore[attr-defined]
        try:
            response = self._transport.get(  # type: ignore[attr-defined]
                url,
                params=params,
                headers={self.AUTH_HEADER: self._api_key},  # type: ignore[attr-defined]
            )
        except Exception as exc:
            self._report_failure(path, f"request failed ({exc})")  # type: ignore[attr-defined]
            return None
        self.quota = QuotaReport.from_headers(  # type: ignore[attr-defined]
            getattr(response, "headers", {}), self.REMAINING_HEADER
        )
        return response

    def _matches_from(self, response: Any, subject: str) -> list[Any] | None:
        """The ``matches`` array, or ``None`` when the body cannot supply one."""
        if response.status_code != self.OK:
            self._report_failure(subject, f"HTTP {response.status_code}")  # type: ignore[attr-defined]
            return None
        try:
            body = response.json()
        except Exception as exc:
            self._report_failure(subject, f"unreadable response ({exc})")  # type: ignore[attr-defined]
            return None
        matches = body.get("matches") if isinstance(body, dict) else None
        return matches if isinstance(matches, list) else None

    def _to_result(self, league: str, match: Any) -> MatchResult | None:
        if not isinstance(match, dict):
            return None
        home = self._team_name(match.get("homeTeam"))
        away = self._team_name(match.get("awayTeam"))
        kickoff = self._parse_time(match.get("utcDate"))
        status = self.STATUS_MAP.get(str(match.get("status") or ""))
        if not home or not away or kickoff is None or status is None:
            return None
        home_goals, away_goals = self._score(match.get("score"), status)
        return MatchResult(
            league=league,
            kickoff=kickoff,
            home_team=home,
            away_team=away,
            status=status,
            home_goals=home_goals,
            away_goals=away_goals,
            # Never derived. The free tier supplies no clock, and computing one
            # from kick-off would ignore stoppages, half-time and delays.
            minute=str(match.get("minute") or ""),
            source=self.name,
            source_id=str(match.get("id") or ""),
        )

    @classmethod
    def _score(cls, score: Any, status: MatchStatus) -> tuple[int | None, int | None]:
        """The current score, or ``(None, None)`` before kick-off.

        ``fullTime`` carries the running score while a match is in play, not
        only the final one — that is how a live match reports 2-0 at 67
        minutes. Before kick-off the API sends nulls, and those stay ``None``
        rather than becoming zeros: 0-0 is a scoreline, and a match that has
        not started does not have one.
        """
        if status is MatchStatus.SCHEDULED or not isinstance(score, dict):
            return (None, None)
        full_time = score.get("fullTime")
        if not isinstance(full_time, dict):
            return (None, None)
        return (cls._as_int(full_time.get("home")), cls._as_int(full_time.get("away")))

    @staticmethod
    def _as_int(value: Any) -> int | None:
        return value if isinstance(value, int) else None

    @staticmethod
    def _team_name(team: Any) -> str:
        """Prefer ``shortName`` — it is far closer to our canonical spellings.

        The API returns ``name`` "FC Porto" and ``shortName`` "Porto"; the
        latter *is* the canonical key here, so preferring it removes an alias
        approval that would otherwise be needed for most clubs.
        """
        if not isinstance(team, dict):
            return ""
        short = str(team.get("shortName") or "").strip()
        return short or str(team.get("name") or "").strip()

    @staticmethod
    def _parse_time(raw: Any) -> datetime | None:
        if not isinstance(raw, str) or not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=None)


class FootballDataHistoryProvider(FootballDataBase, HistoryProvider):
    """Finished matches for one league-season.

    Second in the history chain, behind the local corpus: it answers for the
    current season before football-data.co.uk has published it, and for
    competitions that have no CSV feed at all.
    """

    HISTORY_PATH: ClassVar[str] = "/competitions/{code}/matches"

    def __init__(self, config: Any, transport: Transport, api_key: str) -> None:
        super().__init__(config, transport, api_key)
        #: Competitions this plan has proven it cannot serve.
        self.uncovered: list[str] = []

    @property
    def enabled(self) -> bool:
        return super().enabled and bool(self._api_key)

    def fetch_history(self, query: HistoryQuery) -> list[MatchResult]:
        if not self.enabled or query.league in self.uncovered:
            return []
        code = self._config.competitions.get(query.league)
        if not code:
            # Configuration, not failure: the free tier carries no Scottish,
            # Belgian, Turkish or Greek football at all.
            return []
        year = self._season_year(query.season)
        if year is None:
            self._report_failure(query.league, f"unreadable season {query.season!r}")
            return []

        response = self._get(
            self.HISTORY_PATH.format(code=code),
            {"season": year, "status": "FINISHED"},
        )
        if response is None:
            return []
        if response.status_code in self.NOT_COVERED:
            self.uncovered.append(query.league)
            self._report_failure(
                query.league,
                f"not covered by this plan (HTTP {response.status_code}) — "
                f"skipping it from now on",
            )
            return []

        matches = self._matches_from(response, query.league)
        if not matches:
            # Empty is temporary — a season that has not loaded yet — so the
            # competition is deliberately *not* recorded as uncovered.
            return []
        results = [
            result
            for match in matches
            if (result := self._to_result(query.league, match)) is not None
        ]
        return sorted(results, key=lambda m: m.kickoff)

    @staticmethod
    def _season_year(season: str) -> int | None:
        """``"2526"`` → ``2025``, the year the season started.

        football-data.org identifies a season by its starting year; the rest
        of this project uses football-data.co.uk's ``YYZZ`` form. The
        translation lives here rather than in configuration so a season string
        means one thing everywhere.
        """
        raw = season.strip()
        if len(raw) != 4 or not raw.isdigit():
            return None
        return 2000 + int(raw[:2])


class FootballDataLiveProvider(FootballDataBase, LiveResultsProvider):
    """Every match being played today, across the whole plan, in one request."""

    MATCHES_PATH: ClassVar[str] = "/matches"

    def __init__(
        self,
        config: Any,
        transport: Transport,
        api_key: str,
        today: Callable[[], date] | None = None,
    ) -> None:
        super().__init__(config, transport, api_key)
        #: Injected so a test can pin the day without patching the clock.
        self._today = today or date.today

    @property
    def enabled(self) -> bool:
        return super().enabled and bool(self._api_key)

    def fetch_live(self, leagues: list[str]) -> list[MatchResult]:
        if not self.enabled:
            return []
        # Reverse the configured map once: the response identifies a match by
        # the provider's competition code, and only leagues that were asked
        # for may come back.
        wanted = {
            code: league
            for league, code in self._config.competitions.items()
            if league in leagues
        }
        if not wanted:
            return []

        response = self._get(self.MATCHES_PATH, {"date": self._today().isoformat()})
        if response is None:
            return []
        matches = self._matches_from(response, ",".join(sorted(wanted.values())))
        if not matches:
            return []

        results: list[MatchResult] = []
        for match in matches:
            league = self._league_of(match, wanted)
            if league is None:
                continue
            result = self._to_result(league, match)
            if result is not None:
                results.append(result)
        return sorted(results, key=lambda m: m.kickoff)

    @staticmethod
    def _league_of(match: Any, wanted: dict[str, str]) -> str | None:
        if not isinstance(match, dict):
            return None
        competition = match.get("competition")
        if not isinstance(competition, dict):
            return None
        return wanted.get(str(competition.get("code") or ""))
