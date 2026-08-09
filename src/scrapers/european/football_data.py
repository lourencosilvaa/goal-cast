"""Upcoming fixtures from football-data.org v4.

Call shape from https://www.football-data.org/documentation/quickstart: the key
travels in an ``X-Auth-Token`` header, and matches come from
``/competitions/{code}/matches`` with ``dateFrom``/``dateTo`` filters.

A dormant fallback rather than a live one, for two reasons measured against the
real API:

* **The free tier covers the Champions League only.** Europa returns 403
  ("restricted") and Conference returns 404 ("does not exist"). Neither is a
  fault to report — they are the plan, and repeating them wastes the 10/minute
  allowance — so both are recorded as *uncovered* and skipped thereafter.
* **Its schedule lags a season.** At the time of writing, ``currentSeason`` was
  still 2025-16→2026-05-30 and ``status=SCHEDULED`` returned nothing, so the
  provider legitimately answers empty until 2026-27 loads. That is different
  from being uncovered, and the distinction is kept: an empty CL response must
  not disable the provider for when the data does arrive.
"""

from datetime import datetime
from typing import Any, ClassVar

from src.scrapers.european.providers import (
    EuropeanFixture,
    FixtureProvider,
    FixtureWindow,
    QuotaReport,
)


class FootballDataProvider(FixtureProvider):
    """Reads scheduled matches for the competitions a plan actually covers."""

    name: ClassVar[str] = "football-data.org"

    MATCHES_PATH: ClassVar[str] = "/competitions/{code}/matches"
    AUTH_HEADER: ClassVar[str] = "X-Auth-Token"
    REMAINING_HEADER: ClassVar[str] = "x-requests-available-minute"
    OK: ClassVar[int] = 200
    #: Responses that mean "your plan does not include this", not "it broke".
    NOT_COVERED: ClassVar[tuple[int, ...]] = (403, 404)

    def __init__(self, config: Any, transport: Any, api_key: str) -> None:
        super().__init__(config, transport, api_key)
        #: Competitions this plan has proven it cannot serve.
        self.uncovered: list[str] = []

    def _fetch_one(
        self, competition: str, key: str, window: FixtureWindow
    ) -> list[EuropeanFixture]:
        if competition in self.uncovered:
            return []

        url = f"{self._config.base_url}{self.MATCHES_PATH.format(code=key)}"
        try:
            response = self._transport.get(
                url,
                params={"dateFrom": window.start_iso, "dateTo": window.end_iso},
                headers={self.AUTH_HEADER: self._api_key},
            )
        except Exception as exc:
            self._report_failure(competition, f"request failed ({exc})")
            return []

        self.quota = QuotaReport.from_headers(response.headers, self.REMAINING_HEADER)

        if response.status_code in self.NOT_COVERED:
            self.uncovered.append(competition)
            self._report_failure(
                competition,
                f"not covered by this plan (HTTP {response.status_code}) — "
                f"skipping it from now on",
            )
            return []

        if response.status_code != self.OK:
            self._report_failure(competition, f"HTTP {response.status_code}")
            return []

        try:
            body = response.json()
        except Exception as exc:
            self._report_failure(competition, f"unreadable response ({exc})")
            return []

        matches = body.get("matches") if isinstance(body, dict) else None
        if not isinstance(matches, list):
            return []

        return [
            fixture
            for match in matches
            if (fixture := self._to_fixture(competition, match, window)) is not None
        ]

    def _to_fixture(
        self, competition: str, match: Any, window: FixtureWindow
    ) -> EuropeanFixture | None:
        if not isinstance(match, dict):
            return None
        home = self._team_name(match.get("homeTeam"))
        away = self._team_name(match.get("awayTeam"))
        kickoff = self._parse_time(match.get("utcDate"))
        if not home or not away or kickoff is None:
            return None
        if not window.contains(kickoff):
            return None
        return EuropeanFixture(
            competition=competition,
            kickoff=kickoff,
            home_team=home,
            away_team=away,
            source=self.name,
            source_id=str(match.get("id") or ""),
        )

    @staticmethod
    def _team_name(team: Any) -> str:
        """Prefer ``shortName``: it is far closer to our canonical spellings.

        football-data returns ``name`` "Arsenal FC" and ``shortName`` "Arsenal";
        the latter *is* the canonical key here, so preferring it removes an
        alias approval that would otherwise be needed for every English club.
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
