"""Upcoming fixtures from The Odds API.

Call shape copied from the official samples at
https://github.com/the-odds-api/samples-python — same endpoint, same query
parameter name (``api_key``, not ``apiKey``, though the service accepts both),
same quota headers. The HTTP call is made here by ``requests`` through the
injected transport; no vendor package is involved.

This is the primary provider for one measured reason: ``/events`` costs **zero
credits**. Verified against the live API — fetching events left
``x-requests-used`` at 1 and ``x-requests-remaining`` at 499. Only ``/odds``
draws down the 500/month budget, at one credit per region per market.

Its limitation is inherent rather than contractual: it lists the matches
bookmakers are pricing, so the horizon is roughly the next week or two, and a
competition that has not kicked off yet returns an empty list.
"""

from datetime import datetime
from typing import Any, ClassVar

from src.scrapers.european.providers import (
    EuropeanFixture,
    FixtureProvider,
    FixtureWindow,
    QuotaReport,
)


class OddsApiProvider(FixtureProvider):
    """Lists events for a sport key, in the provider's own team spelling."""

    name: ClassVar[str] = "the-odds-api"

    #: Free endpoint. ``/odds`` is the one that costs credits.
    EVENTS_PATH: ClassVar[str] = "/sports/{key}/events"
    #: samples-python passes the key as a query parameter under this name.
    API_KEY_PARAM: ClassVar[str] = "api_key"
    REMAINING_HEADER: ClassVar[str] = "x-requests-remaining"
    USED_HEADER: ClassVar[str] = "x-requests-used"
    OK: ClassVar[int] = 200

    def _fetch_one(
        self, competition: str, key: str, window: FixtureWindow
    ) -> list[EuropeanFixture]:
        url = f"{self._config.base_url}{self.EVENTS_PATH.format(key=key)}"
        try:
            response = self._transport.get(
                url, params={self.API_KEY_PARAM: self._api_key}
            )
        except Exception as exc:
            self._report_failure(competition, f"request failed ({exc})")
            return []

        self.quota = QuotaReport.from_headers(
            response.headers, self.REMAINING_HEADER, self.USED_HEADER
        )

        if response.status_code != self.OK:
            self._report_failure(
                competition, f"HTTP {response.status_code}: {response.text[:120]}"
            )
            return []

        try:
            events = response.json()
        except Exception as exc:
            self._report_failure(competition, f"unreadable response ({exc})")
            return []
        if not isinstance(events, list):
            return []

        return [
            fixture
            for event in events
            if (fixture := self._to_fixture(competition, event, window)) is not None
        ]

    def _to_fixture(
        self, competition: str, event: Any, window: FixtureWindow
    ) -> EuropeanFixture | None:
        if not isinstance(event, dict):
            return None
        home = str(event.get("home_team") or "").strip()
        away = str(event.get("away_team") or "").strip()
        kickoff = self._parse_time(event.get("commence_time"))
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
            source_id=str(event.get("id") or ""),
        )

    @staticmethod
    def _parse_time(raw: Any) -> datetime | None:
        """ISO-8601 UTC, e.g. ``2026-08-11T15:00:00Z``.

        Returned naive in UTC: the rest of the pipeline keys fixtures by
        calendar date, and mixing aware and naive datetimes downstream is a
        reliable source of silent comparison bugs.
        """
        if not isinstance(raw, str) or not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=None)
