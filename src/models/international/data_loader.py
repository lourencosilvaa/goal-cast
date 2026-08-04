from pathlib import Path

import pandas as pd

from config.config_loader import InternationalConfig
from src.models.international.base import AbstractMatchDataLoader

# Raw Kaggle column -> canonical column
_COLUMN_MAP = {
    "date": "Date",
    "home_team": "HomeTeam",
    "away_team": "AwayTeam",
    "home_score": "FTHG",
    "away_score": "FTAG",
    "tournament": "Tournament",
}


class InternationalDataLoader(AbstractMatchDataLoader):
    """Loads the fixed Kaggle international results dataset.

    Normalizes the raw ``results.csv`` (date, home_team, away_team,
    home_score, away_score, tournament, city, country, neutral) into the
    canonical schema (``Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR``) and adds
    the international-specific ``Neutral`` and ``Tournament`` columns. Applies
    the configured ``min_date`` and ``tournaments`` filters.
    """

    def __init__(self, config: InternationalConfig) -> None:
        self.config = config

    def _dataset_path(self) -> Path:
        return Path(self.config.dataset_path)

    @staticmethod
    def _derive_result(home_goals: int, away_goals: int) -> str:
        if home_goals > away_goals:
            return "H"
        if home_goals < away_goals:
            return "A"
        return "D"

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns=_COLUMN_MAP).copy()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])

        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)
        df["FTR"] = [self._derive_result(h, a) for h, a in zip(df["FTHG"], df["FTAG"])]
        if "neutral" in df.columns:
            df["Neutral"] = self._parse_neutral(df["neutral"])
        else:
            df["Neutral"] = False

        keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Neutral"]
        if "Tournament" in df.columns:
            keep.append("Tournament")
        return df[keep].reset_index(drop=True)

    @staticmethod
    def _parse_neutral(series: pd.Series) -> pd.Series:
        truthy = {"true", "1", "yes", "t"}
        return series.astype(str).str.strip().str.lower().isin(truthy)

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.min_date:
            df = df[df["Date"] >= pd.Timestamp(self.config.min_date)]
        if self.config.tournaments and "Tournament" in df.columns:
            df = df[df["Tournament"].isin(self.config.tournaments)]
        return df.sort_values("Date").reset_index(drop=True)

    def load_all(self) -> pd.DataFrame:
        """Return all matches in the canonical schema, or empty if missing."""
        path = self._dataset_path()
        if not path.exists():
            print(f"International dataset not found: {path}")
            return pd.DataFrame()
        try:
            raw = pd.read_csv(path, encoding="utf-8")
        except Exception as exc:
            print(f"Error reading international dataset {path}: {exc}")
            return pd.DataFrame()

        normalized = self._normalize(raw)
        if normalized.empty:
            return normalized
        return self._apply_filters(normalized)
