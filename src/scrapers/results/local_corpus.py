"""Finished matches from the football-data.co.uk season CSVs.

First in the history chain, and the reason history is effectively free: the
training cache already holds every season of every tracked division, so the
common case answers from disk with no request, no quota and no API key. It is
also the *same* data the models were trained on, which is worth more than it
sounds — a results page disagreeing with the corpus behind a prediction would
be a bug that is very hard to see.

**Parsed with the standard library, deliberately.** The rest of this project
reads these files with pandas (``src/models/data_loader.py``), but that loader
lives in the ML image; this provider runs inside the results service, whose
dependency group is fastapi/requests/playwright and nothing scientific. Adding
pandas here to save twenty lines of ``csv`` would put NumPy in a container
whose entire job is to answer two HTTP endpoints.

Downloading is the fallback, not the path: a deployed service has no training
cache, so the first request for a season fetches it and writes it into
``results.cache_dir``, after which that season behaves like a local file.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Iterable

from src.scrapers.results.base import HistoryProvider, Transport
from src.scrapers.results.models import HistoryQuery, MatchResult, MatchStatus


class LocalCorpusHistoryProvider(HistoryProvider):
    """Reads (and, when absent, fetches) one ``{season}_{league}.csv``."""

    name: ClassVar[str] = "local-corpus"

    #: Columns a row must supply to become a match. The rest of the file is
    #: odds and match statistics that a results view has no use for.
    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
    )
    #: Date spellings football-data.co.uk has used. Recent seasons write
    #: four-digit years, older ones two.
    DATE_FORMATS: ClassVar[tuple[str, ...]] = ("%d/%m/%Y", "%d/%m/%y")
    TIME_FORMAT: ClassVar[str] = "%H:%M"
    #: Some season files carry a stray cp1252 byte inside an odds column, and
    #: a strict UTF-8 read aborts on it and loses the whole season. latin-1
    #: decodes every byte, so it is the safe last resort — the same ladder
    #: ``FootballDataLoader`` climbs, for the same measured reason.
    ENCODINGS: ClassVar[tuple[str, ...]] = ("utf-8", "latin-1")

    def __init__(
        self,
        config: Any,
        transport: Transport,
        cache_dir: str,
        api_key: str = "",
    ) -> None:
        super().__init__(config, transport, api_key)
        self._cache_dir = Path(cache_dir)

    def fetch_history(self, query: HistoryQuery) -> list[MatchResult]:
        if not self.enabled:
            return []
        if query.league not in self._config.leagues:
            # No football-data.co.uk feed exists for this code — the UEFA
            # competitions are the reason the openfootball corpus exists.
            # Requesting one would 404 on every history call.
            return []
        text = self._read(query) or self._download(query)
        if text is None:
            return []
        return sorted(self._parse(text, query.league), key=lambda m: m.kickoff)

    # ── locating the file ────────────────────────────────────────────────

    def _filename(self, query: HistoryQuery) -> str:
        return f"{query.season}_{query.league}.csv"

    def _candidates(self, query: HistoryQuery) -> Iterable[Path]:
        name = self._filename(query)
        for directory in self._config.search_dirs:
            yield Path(directory) / name
        yield self._cache_dir / name

    def _read(self, query: HistoryQuery) -> str | None:
        for path in self._candidates(query):
            if not path.is_file():
                continue
            for encoding in self.ENCODINGS:
                try:
                    return path.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    continue
        return None

    def _download(self, query: HistoryQuery) -> str | None:
        url = f"{self._config.base_url}/{query.season}/{query.league}.csv"
        try:
            response = self._transport.get(url)
        except Exception as exc:
            self._report_failure(query.league, f"download failed ({exc})")
            return None
        if getattr(response, "status_code", 0) != 200:
            self._report_failure(
                query.league,
                f"{query.season}: HTTP {getattr(response, 'status_code', '?')} "
                f"from {url}",
            )
            return None
        text = str(getattr(response, "text", "") or "")
        if not text.strip():
            return None
        self._cache(query, text)
        return text

    def _cache(self, query: HistoryQuery, text: str) -> None:
        """Persist a downloaded season so it is fetched exactly once.

        Failing to write is not failing to answer: a read-only or full disk
        costs a repeated download, which is worth reporting and not worth
        turning into an error for the caller.
        """
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            (self._cache_dir / self._filename(query)).write_text(text)
        except OSError as exc:
            self._report_failure(query.league, f"could not cache season ({exc})")

    # ── parsing ──────────────────────────────────────────────────────────

    def _parse(self, text: str, league: str) -> list[MatchResult]:
        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames:
            return []
        missing = [c for c in self.REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            self._report_failure(league, f"missing columns: {', '.join(missing)}")
            return []
        return [
            match
            for row in reader
            if (match := self._to_match(row, league)) is not None
        ]

    def _to_match(self, row: dict, league: str) -> MatchResult | None:
        kickoff = self._kickoff(row.get("Date"), row.get("Time"))
        home = str(row.get("HomeTeam") or "").strip()
        away = str(row.get("AwayTeam") or "").strip()
        home_goals = self._goals(row.get("FTHG"))
        away_goals = self._goals(row.get("FTAG"))
        if kickoff is None or not home or not away:
            return None
        if home_goals is None or away_goals is None:
            # A row with no full-time score is an abandoned or not-yet-played
            # fixture. Skipped rather than filled in with zeros.
            return None
        return MatchResult(
            league=league,
            kickoff=kickoff,
            home_team=home,
            away_team=away,
            status=MatchStatus.FINISHED,
            home_goals=home_goals,
            away_goals=away_goals,
            source=self.name,
        )

    @classmethod
    def _kickoff(cls, raw_date: Any, raw_time: Any) -> datetime | None:
        day = cls._parse_date(str(raw_date or "").strip())
        if day is None:
            return None
        clock = cls._parse_clock(str(raw_time or "").strip())
        if clock is None:
            # Seasons before 2019 carry no Time column at all. Midnight is
            # visibly a placeholder; inventing a plausible evening kick-off
            # would not be.
            return day
        return day.replace(hour=clock[0], minute=clock[1])

    @classmethod
    def _parse_date(cls, raw: str) -> datetime | None:
        for fmt in cls.DATE_FORMATS:
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        return None

    @classmethod
    def _parse_clock(cls, raw: str) -> tuple[int, int] | None:
        try:
            parsed = datetime.strptime(raw, cls.TIME_FORMAT)
        except ValueError:
            return None
        return (parsed.hour, parsed.minute)

    @staticmethod
    def _goals(raw: Any) -> int | None:
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            return None
