"""Where a live snapshot survives a restart.

An interface plus a JSON-on-disk implementation. The interface is what makes
the storage choice reversible — Redis, a Supabase table, or nothing at all are
all substitutions rather than rewrites — and the JSON one is chosen because
the volume is one small file per league set and the service already has a
cache directory.

**This is a cache, not a record.** Every failure path returns "no snapshot"
rather than raising: a corrupt file, a full disk or a read-only mount must
cost one extra fetch, never a failed request. Losing it entirely costs
nothing but a cold board on the first request after a restart.
"""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Sequence

from src.scrapers.results.models import LiveSnapshot, MatchResult, MatchStatus


class ResultsRepository(ABC):
    """Storage for the most recent snapshot of a set of leagues."""

    @abstractmethod
    def load(self, key: Sequence[str]) -> LiveSnapshot | None:
        """The stored snapshot for ``key``, or ``None`` if there is none."""

    @abstractmethod
    def save(self, key: Sequence[str], snapshot: LiveSnapshot) -> None:
        """Store ``snapshot`` under ``key``, replacing any predecessor."""


class JsonResultsRepository(ResultsRepository):
    """One JSON file per league set, under the configured cache directory."""

    PREFIX: ClassVar[str] = "live_"
    SUFFIX: ClassVar[str] = ".json"
    #: Filename for the "every league" request, which has an empty key.
    ALL: ClassVar[str] = "all"
    #: League codes reach us from a query string, so the filename is built
    #: from a whitelist rather than by escaping what arrives: a code of
    #: "../../etc/passwd" must not be able to name a file anywhere else.
    SAFE: ClassVar[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9]+")

    def __init__(self, directory: str) -> None:
        self._directory = Path(directory)

    def path_for(self, key: Sequence[str]) -> Path:
        parts = [cleaned for code in key if (cleaned := self.SAFE.sub("", code))]
        name = "-".join(parts) if parts else self.ALL
        return self._directory / f"{self.PREFIX}{name}{self.SUFFIX}"

    def load(self, key: Sequence[str]) -> LiveSnapshot | None:
        path = self.path_for(key)
        try:
            body = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return self._to_snapshot(body)

    def save(self, key: Sequence[str], snapshot: LiveSnapshot) -> None:
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            self.path_for(key).write_text(json.dumps(self._to_json(snapshot)))
        except OSError as exc:
            print(f"[results:repository] could not persist snapshot ({exc})")

    # ── serialisation ────────────────────────────────────────────────────

    @staticmethod
    def _to_json(snapshot: LiveSnapshot) -> dict:
        return {
            "fetched_at": snapshot.fetched_at.isoformat(),
            "source": snapshot.source,
            "matches": [
                {
                    "league": match.league,
                    "kickoff": match.kickoff.isoformat(),
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "status": match.status.value,
                    "home_goals": match.home_goals,
                    "away_goals": match.away_goals,
                    "minute": match.minute,
                    "source": match.source,
                    "source_id": match.source_id,
                }
                for match in snapshot.matches
            ],
        }

    @classmethod
    def _to_snapshot(cls, body: Any) -> LiveSnapshot | None:
        if not isinstance(body, dict):
            return None
        fetched_at = cls._parse_time(body.get("fetched_at"))
        raw_matches = body.get("matches")
        if fetched_at is None or not isinstance(raw_matches, list):
            return None
        matches = tuple(
            match for raw in raw_matches if (match := cls._to_match(raw)) is not None
        )
        return LiveSnapshot(
            fetched_at=fetched_at,
            matches=matches,
            source=str(body.get("source") or ""),
        )

    @classmethod
    def _to_match(cls, raw: Any) -> MatchResult | None:
        """One match, or ``None`` if the record cannot be trusted.

        Dropping an unreadable match rather than the whole file: a snapshot
        that lost one row is still a better board than no board.
        """
        if not isinstance(raw, dict):
            return None
        kickoff = cls._parse_time(raw.get("kickoff"))
        try:
            status = MatchStatus(str(raw.get("status")))
        except ValueError:
            return None
        league = str(raw.get("league") or "")
        home = str(raw.get("home_team") or "")
        away = str(raw.get("away_team") or "")
        if kickoff is None or not league or not home or not away:
            return None
        return MatchResult(
            league=league,
            kickoff=kickoff,
            home_team=home,
            away_team=away,
            status=status,
            home_goals=cls._as_int(raw.get("home_goals")),
            away_goals=cls._as_int(raw.get("away_goals")),
            minute=str(raw.get("minute") or ""),
            source=str(raw.get("source") or ""),
            source_id=str(raw.get("source_id") or ""),
        )

    @staticmethod
    def _parse_time(raw: Any) -> datetime | None:
        if not isinstance(raw, str):
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    @staticmethod
    def _as_int(value: Any) -> int | None:
        return value if isinstance(value, int) else None
