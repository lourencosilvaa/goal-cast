"""Parser for the openfootball ``football.txt`` match format.

The format is human-written plain text, and three of its conventions are
actively dangerous to a naive reader:

**Scorelines put the decisive number first.** A normal match reads
``0-3 (0-2)`` — full time, then half time. But an extra-time match reads
``4-1 a.e.t. (1-1, 0-1)``, where ``4-1`` is the score *after extra time* and
the bracket is ``(90 minutes, half time)``. With penalties it grows a third
group at the front: ``3-4 pen. 1-1 a.e.t. (1-1, 0-0)``. Reading the first pair
would record a penalty shootout as a 3-4 defeat.

This parser keeps the **90-minute** score as ``FTHG``/``FTAG``, which is what
football-data.co.uk means by full time. Domestic and European rows then carry
the same meaning, and a tie settled on penalties correctly enters the ratings
as the draw it was.

**Dates carry a year only sometimes, and the calendar restarts.** A bare
``Wed Jan 15`` has no year of its own, and it is tempting to infer one from
the previous line. That fails: openfootball lists each group in full before
starting the next, so a bare ``Wed Sep 14`` regularly follows a ``Wed Dec 7``.
Treating that as "the month went backwards, so the year advanced" walks the
2011-12 season forward into 2018. The season's own two calendar years are the
only safe authority, so a bare date is placed by its month — autumn in the
first year, spring in the second — and an explicit year always wins.

**Times are inherited.** A match listed without a kick-off time belongs to the
same slot as the line above it.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, ClassVar, Optional

import pandas as pd


@dataclass(frozen=True)
class SeasonSpec:
    """Which competition-season a file holds.

    Bundled rather than passed as three parameters (§4.3), and carrying
    ``years`` because a bare date line cannot be placed without them.

    Attributes:
        competition: Code stamped onto every row's ``Div`` (e.g. ``CL``).
        season: Season label carried through for traceability (``2024-25``).
        years: The season's two calendar years, e.g. ``(2024, 2025)``.
    """

    competition: str
    season: str
    years: tuple[int, int]


class OpenFootballParser:
    """Turns one ``football.txt`` file into goals-only match rows."""

    #: A match line: optional kick-off, two sides split by a lone ``v``, then
    #: whatever score notation follows after at least two spaces.
    MATCH_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*(?:(?P<time>\d{1,2}:\d{2})\s+)?"
        r"(?P<home>\S.*?)\s+v\s+(?P<away>\S.*?)"
        r"(?:\s{2,}(?P<score>.*))?$"
    )

    #: A date line, e.g. ``Tue Sep 17 2024`` or ``Wed Sep 18``.
    DATE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*[A-Z][a-z]{2}\s+(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})"
        r"(?:\s+(?P<year>\d{4}))?\s*$"
    )

    #: A trailing ``(POR)``-style country code on a team name.
    COUNTRY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<name>.*?)\s*\((?P<code>[A-Z]{3})\)$"
    )

    #: Every score group in a line, e.g. ``1-1`` — ordered as written.
    SCORE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"(\d+)\s*-\s*(\d+)")

    #: The bracketed group holding regulation scores.
    BRACKET_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\(([^)]*)\)")

    #: Marks a tie that ran past 90 minutes, moving the regulation score into
    #: the bracket.
    EXTRA_TIME_MARKER: ClassVar[str] = "a.e.t."

    #: Lines carrying one of these are not football results and are dropped.
    #: ``awarded`` covers forfeits (a 3-0 nobody played); ``cancelled`` and
    #: ``abandoned`` speak for themselves.
    EXCLUDED_MARKERS: ClassVar[tuple[str, ...]] = ("awarded", "cancelled", "abandoned")

    #: Lines that introduce structure rather than a match.
    IGNORED_PREFIXES: ClassVar[tuple[str, ...]] = ("=", "#", "▪")

    MONTHS: ClassVar[dict[str, int]] = {
        month: number
        for number, month in enumerate(
            (
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ),
            start=1,
        )
    }

    #: First month of a European season. A bare date at or after this belongs
    #: to the season's first calendar year, anything earlier to its second.
    SEASON_START_MONTH: ClassVar[int] = 7

    def parse(self, text: str, spec: SeasonSpec) -> list[dict[str, Any]]:
        """Parse one competition-season file into match rows."""
        rows: list[dict[str, Any]] = []
        current: Optional[date] = None

        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith(self.IGNORED_PREFIXES):
                continue

            parsed_date = self._parse_date(line, spec)
            if parsed_date is not None:
                current = parsed_date
                continue

            row = self._parse_match(line, current, spec)
            if row is not None:
                rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    def _parse_date(self, line: str, spec: SeasonSpec) -> Optional[date]:
        """A date line's date, or ``None`` when the line is not one."""
        match = self.DATE_PATTERN.match(line)
        if match is None:
            return None

        month = self.MONTHS.get(match.group("month"))
        if month is None:
            return None

        year_text = match.group("year")
        year = (
            int(year_text) if year_text is not None else self._season_year(month, spec)
        )

        try:
            return date(year, month, int(match.group("day")))
        except ValueError:
            return None

    def _season_year(self, month: int, spec: SeasonSpec) -> int:
        """The calendar year a bare ``Wed Jan 15`` belongs to.

        Derived from the season rather than from the previous date line, which
        is what makes it immune to the per-group calendar restarts.
        """
        start_year, end_year = spec.years
        return start_year if month >= self.SEASON_START_MONTH else end_year

    # ------------------------------------------------------------------
    # Matches
    # ------------------------------------------------------------------

    def _parse_match(
        self,
        line: str,
        current: Optional[date],
        spec: SeasonSpec,
    ) -> Optional[dict[str, Any]]:
        if current is None:
            return None  # a match with no date cannot be placed in time

        match = self.MATCH_PATTERN.match(line)
        if match is None:
            return None

        score = (match.group("score") or "").strip()
        if not score or self._is_excluded(score):
            return None

        goals = self._scores(score)
        if goals is None:
            return None
        full_time, half_time = goals

        home, home_country = self._split_country(match.group("home"))
        away, away_country = self._split_country(match.group("away"))
        if not home or not away:
            return None

        return {
            "Div": spec.competition,
            "Season": spec.season,
            "Date": pd.Timestamp(current),
            "HomeTeam": home,
            "AwayTeam": away,
            "HomeCountry": home_country,
            "AwayCountry": away_country,
            "FTHG": full_time[0],
            "FTAG": full_time[1],
            "HTHG": half_time[0] if half_time else None,
            "HTAG": half_time[1] if half_time else None,
        }

    def _is_excluded(self, score: str) -> bool:
        lowered = score.lower()
        return any(marker in lowered for marker in self.EXCLUDED_MARKERS)

    def _scores(
        self, score: str
    ) -> Optional[tuple[tuple[int, int], Optional[tuple[int, int]]]]:
        """``(90-minute, half-time)`` goals from a score notation.

        Read structurally rather than by counting number pairs, because the
        notation grows a group at a time as a tie goes deeper and every one of
        these six shapes occurs in the shipped corpus::

            0-3 (0-2)                          full time (half time)
            0-0                                full time only
            4-1 a.e.t. (1-1, 0-1)              extra time (90, half time)
            4-1 a.e.t. (1-1)                   extra time (90)
            3-4 pen. 1-1 a.e.t. (1-1, 0-0)     penalties, extra time (90, ht)
            4-2 pen. 1-1 a.e.t. (1-1)          penalties, extra time (90)

        Once extra time is involved the bracket — not the leading pair — holds
        the 90-minute result, so a shootout can never be mistaken for the
        score. Half time is genuinely absent in some rows and stays ``None``
        rather than being invented.
        """
        bracket_match = self.BRACKET_PATTERN.search(score)
        bracket = self._pairs(bracket_match.group(1)) if bracket_match else []
        outside = self._pairs(
            score[: bracket_match.start()] if bracket_match else score
        )

        if self.EXTRA_TIME_MARKER in score:
            # The tie ran long: the bracket holds 90 minutes, then half time.
            if not bracket:
                return None
            return bracket[0], bracket[1] if len(bracket) > 1 else None

        if not outside:
            return None
        return outside[0], bracket[0] if bracket else None

    def _pairs(self, text: str) -> list[tuple[int, int]]:
        return [
            (int(home), int(away)) for home, away in self.SCORE_PATTERN.findall(text)
        ]

    def _split_country(self, raw: str) -> tuple[str, Optional[str]]:
        """A team's name and its country code, which is kept for matching.

        The code is what makes an automated name suggestion safe to show: an
        openfootball name is otherwise close enough to the wrong club to be
        dangerous ("AC Sparta Praha" scores well against "Sparta Rotterdam").
        """
        text = raw.strip()
        match = self.COUNTRY_PATTERN.match(text)
        if match is None:
            return text, None
        return match.group("name").strip(), match.group("code")
