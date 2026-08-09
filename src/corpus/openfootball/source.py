"""Corpus source reading a local openfootball checkout.

Reads only from disk — the checkout is refreshed by ``OpenFootballRepository``
as a separate, explicit step — so this can run inside a corpus build with no
network at all.

Coverage in the upstream repository is uneven and worth knowing before reading
the output: Champions League runs from 2011-12, Europa League only from
2020-21, Conference League from 2021-22, and qualifiers only from 2024-25. A
configured season whose file does not exist is skipped silently, because a
missing competition-season is the normal case, not an error.
"""

from pathlib import Path
from typing import ClassVar, Optional

import pandas as pd

from config.config_loader import EuropeanConfig
from src.corpus.openfootball.parser import OpenFootballParser, SeasonSpec
from src.corpus.openfootball.repository import OpenFootballRepository
from src.corpus.supplementary import CorpusSchema, SupplementaryCorpusSource


class OpenFootballCorpusSource(SupplementaryCorpusSource):
    """Parses the checked-out openfootball files into a goals-only corpus.

    The repository and parser are injected so the source is testable without
    network access and the upstream project stays a replaceable detail.
    """

    #: Length of a football-data season code, e.g. ``2425``.
    SEASON_CODE_LENGTH: ClassVar[int] = 4

    #: Two-digit years at or above this belong to the twentieth century.
    CENTURY_PIVOT: ClassVar[int] = 90

    #: openfootball names its season directories ``2024-25``.
    SEASON_DIRECTORY_FORMAT: ClassVar[str] = "{start}-{end:02d}"

    FILE_SUFFIX: ClassVar[str] = ".txt"
    ENCODING: ClassVar[str] = "utf-8"

    def __init__(
        self,
        config: EuropeanConfig,
        repository: OpenFootballRepository,
        parser: Optional[OpenFootballParser] = None,
        include_qualifiers: bool = False,
    ) -> None:
        self.config = config
        self._repository = repository
        self._parser = parser or OpenFootballParser()
        self._include_qualifiers = include_qualifiers

    # ------------------------------------------------------------------
    # Season naming
    # ------------------------------------------------------------------

    def season_years(self, season: str) -> tuple[int, int]:
        """Expand a ``YYZZ`` season code into its two calendar years."""
        text = season.strip()
        if len(text) != self.SEASON_CODE_LENGTH or not text.isdigit():
            raise ValueError(
                f"Season must be a {self.SEASON_CODE_LENGTH}-digit code "
                f"like '2425', got {season!r}"
            )
        return self._full_year(int(text[:2])), self._full_year(int(text[2:]))

    def _full_year(self, two_digit: int) -> int:
        century = 1900 if two_digit >= self.CENTURY_PIVOT else 2000
        return century + two_digit

    def season_directory(self, season: str) -> str:
        """The openfootball directory name for a season code."""
        start_year, end_year = self.season_years(season)
        return self.SEASON_DIRECTORY_FORMAT.format(start=start_year, end=end_year % 100)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _competitions(self) -> dict[str, str]:
        """Competition code → file stem, qualifiers included only on request."""
        competitions = dict(self.config.competitions)
        if self._include_qualifiers:
            competitions.update(self.config.qualifier_competitions)
        return competitions

    def load(self) -> pd.DataFrame:
        """Every configured competition-season present in the checkout."""
        if not self.config.enabled:
            return CorpusSchema.empty()

        root = self._repository.path
        if not root.is_dir():
            return CorpusSchema.empty()

        rows: list[dict[str, object]] = []
        competitions = self._competitions()
        for season in self.config.seasons:
            try:
                directory = root / self.season_directory(season)
                years = self.season_years(season)
            except ValueError:
                continue
            if not directory.is_dir():
                continue
            for code, stem in competitions.items():
                rows.extend(self._read(directory, stem, code, season, years))

        return CorpusSchema.from_rows(rows)

    def _read(
        self,
        directory: Path,
        stem: str,
        code: str,
        season: str,
        years: tuple[int, int],
    ) -> list[dict[str, object]]:
        """One competition-season file, or nothing when it is absent.

        A missing file is the normal case — Europa League simply does not
        exist before 2020-21 — so it is not reported as a failure.
        """
        path = directory / f"{stem}{self.FILE_SUFFIX}"
        if not path.is_file():
            return []
        try:
            text = path.read_text(encoding=self.ENCODING)
        except (OSError, UnicodeDecodeError):
            return []
        spec = SeasonSpec(competition=code, season=season, years=years)
        return self._parser.parse(text, spec)
