"""Goals-only match history that supplements the domestic corpus.

The domestic corpus (football-data.co.uk) ships shots, fouls and corners, and
``FeatureEngineer.build_match_features`` needs them — a row without those
columns is dropped, which is why ``EC`` (National League) is excluded from
``data.leagues``. UEFA club competitions have no such feed, so their results
can never join the ensemble's feature matrix.

They do not need to. The reason European results matter is *calibration*: each
domestic league is a near-closed pool, so ELO ratings and Dixon-Coles
attack/defence strengths float independently per league and are not comparable
across them. Only real cross-league matches link the pools — and both ELO and
Dixon-Coles need nothing but goals.

This module defines the narrow contract those matches arrive under. Sources
are interchangeable by design (§4.5): openfootball today, an API or the user's
own database later, with nothing downstream aware of the swap.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Sequence

import numpy as np
import pandas as pd


class CorpusSchema:
    """The canonical shape every supplementary source must produce.

    ``normalise`` is total: any frame goes in, a conforming frame comes out.
    Rows that cannot be trusted — unparseable date, missing team, missing
    score — are dropped rather than repaired, because a fabricated result
    would corrupt the very ratings this corpus exists to calibrate.
    """

    #: Column order every consumer may rely on. ``HTHG``/``HTAG`` and the
    #: country codes are not needed for calibration, but the source provides
    #: them for free: half-time goals match the domestic schema, and the
    #: country code is what makes a later name suggestion safe to offer.
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "Div",
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "HomeCountry",
        "AwayCountry",
        "FTHG",
        "FTAG",
        "FTR",
        "HTHG",
        "HTAG",
    )

    #: Columns a source must supply; everything else is derived or optional.
    REQUIRED: ClassVar[tuple[str, ...]] = (
        "Div",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
    )

    #: Columns identifying one match, for de-duplication across sources.
    IDENTITY: ClassVar[tuple[str, ...]] = ("Div", "Date", "HomeTeam", "AwayTeam")

    #: Present when the source supplies them, all-null otherwise.
    OPTIONAL: ClassVar[tuple[str, ...]] = (
        "Season",
        "HomeCountry",
        "AwayCountry",
        "HTHG",
        "HTAG",
    )

    HOME_WIN: ClassVar[str] = "H"
    DRAW: ClassVar[str] = "D"
    AWAY_WIN: ClassVar[str] = "A"
    VALID_RESULTS: ClassVar[tuple[str, ...]] = (HOME_WIN, DRAW, AWAY_WIN)

    @classmethod
    def empty(cls) -> pd.DataFrame:
        """An empty frame carrying the canonical columns."""
        return pd.DataFrame({column: [] for column in cls.COLUMNS})

    @classmethod
    def from_rows(cls, rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
        """Normalise parser output (a list of row dicts)."""
        if not rows:
            return cls.empty()
        return cls.normalise(pd.DataFrame(list(rows)))

    @classmethod
    def normalise(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce any frame into the canonical shape, dropping untrusted rows."""
        if df is None or df.empty:
            return cls.empty()
        if any(column not in df.columns for column in cls.REQUIRED):
            return cls.empty()

        frame = df.copy()
        for column in cls.OPTIONAL:
            if column not in frame.columns:
                frame[column] = pd.NA

        frame["Date"] = cls._dates(frame["Date"])
        frame = frame.dropna(subset=["Date"])

        for column in ("HomeTeam", "AwayTeam"):
            frame[column] = frame[column].astype("string").str.strip()
            frame = frame[frame[column].notna() & (frame[column] != "")]

        for column in ("FTHG", "FTAG"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["FTHG", "FTAG"])
        if frame.empty:
            return cls.empty()

        frame["FTHG"] = frame["FTHG"].astype(int)
        frame["FTAG"] = frame["FTAG"].astype(int)
        for column in ("HTHG", "HTAG"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["Div"] = frame["Div"].astype("string").str.strip()
        frame["FTR"] = cls._results(frame)

        return frame[list(cls.COLUMNS)].reset_index(drop=True)

    @staticmethod
    def _dates(values: pd.Series) -> pd.Series:
        """Parse dates per element, coercing anything unparseable to ``NaT``.

        The format is declared ``mixed`` explicitly rather than left to
        pandas' inference, which warns when it gives up.
        """
        if pd.api.types.is_datetime64_any_dtype(values):
            return values
        parsed: pd.Series = pd.to_datetime(values, errors="coerce", format="mixed")
        return parsed

    @classmethod
    def _results(cls, frame: pd.DataFrame) -> pd.Series:
        """Full-time results, derived from the 90-minute goals.

        A supplied ``FTR`` is kept: a source that reports a result may know
        about an award or annulment the scoreline does not show.
        """
        derived = pd.Series(
            np.where(
                frame["FTHG"] > frame["FTAG"],
                cls.HOME_WIN,
                np.where(frame["FTHG"] < frame["FTAG"], cls.AWAY_WIN, cls.DRAW),
            ),
            index=frame.index,
            dtype="object",
        )
        if "FTR" not in frame.columns:
            return derived
        supplied = frame["FTR"].astype("string").str.strip().str.upper()
        return supplied.where(supplied.isin(list(cls.VALID_RESULTS)), derived).astype(
            "object"
        )


class SupplementaryCorpusSource(ABC):
    """A source of goals-only match history."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return matches in the :class:`CorpusSchema` shape, or an empty frame.

        Implementations degrade rather than raise: an unavailable source must
        never take a training run down.
        """


class ChainedCorpusSource(SupplementaryCorpusSource):
    """Every match from every source, in order.

    Sources accumulate rather than override — a fallback provider may hold
    seasons the primary lacks — and a source that fails contributes nothing,
    mirroring ``ChainedTeamAliasRepository``.
    """

    def __init__(self, sources: Sequence[SupplementaryCorpusSource]) -> None:
        self._sources = list(sources)

    def load(self, deduplicate: bool = True) -> pd.DataFrame:
        """Concatenate every source, dropping repeats of the same match.

        De-duplication is on by default here (unlike aliases, which
        accumulate): two sources reporting the same fixture is a duplicate
        result, and feeding it twice would double that match's weight in the
        ratings. The earliest source wins.
        """
        frames: list[pd.DataFrame] = []
        for source in self._sources:
            try:
                frame = CorpusSchema.normalise(source.load())
            except Exception:
                continue
            if not frame.empty:
                frames.append(frame)

        if not frames:
            return CorpusSchema.empty()

        combined = pd.concat(frames, ignore_index=True)
        if deduplicate:
            combined = combined.drop_duplicates(
                subset=list(CorpusSchema.IDENTITY), keep="first"
            ).reset_index(drop=True)
        return combined


class StaticFileCorpusSource(SupplementaryCorpusSource):
    """Reads a parsed corpus back from a CSV or Parquet cache.

    This is what keeps the network out of the training path: the harvest
    writes the cache, and a model refit reads only this.
    """

    READERS: ClassVar[dict[str, str]] = {".csv": "read_csv", ".parquet": "read_parquet"}

    def __init__(self, path: Any) -> None:
        from pathlib import Path

        self._path = Path(path)

    @property
    def path(self) -> Any:
        return self._path

    def load(self) -> pd.DataFrame:
        reader_name: Optional[str] = self.READERS.get(self._path.suffix.lower())
        if reader_name is None or not self._path.is_file():
            return CorpusSchema.empty()
        try:
            reader = getattr(pd, reader_name)
            return CorpusSchema.normalise(reader(self._path))
        except Exception:
            return CorpusSchema.empty()
