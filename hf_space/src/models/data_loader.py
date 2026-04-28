import time
from pathlib import Path
from typing import Optional

import pandas as pd

from config.config_loader import DataConfig, HuggingFaceConfig

_CACHE_DIR = Path("/tmp/datasets/cache")
_CACHE_MAX_AGE = 24 * 60 * 60  # seconds

# Reverse mapping: league name → league code (built at runtime from config)
_LEAGUE_NAME_TO_CODE: dict[str, str] = {}


class FootballDataLoader:
    """
    Historical football match data loader.
    Source priority: HuggingFace Parquet → local CSV cache → football-data.co.uk

    HF Parquet files are stored as one file per league (all seasons merged)
    under {hf_config.local_dir}/{hf_config.dataset_subfolder}/{league_code}.parquet.
    """

    def __init__(
        self,
        config: DataConfig,
        hf_config: Optional[HuggingFaceConfig] = None,
    ) -> None:
        self.config = config
        self.hf_config = hf_config
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        global _LEAGUE_NAME_TO_CODE
        _LEAGUE_NAME_TO_CODE = {v: k for k, v in config.leagues.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cached_path(self, league: str, season: str) -> Path:
        return _CACHE_DIR / f"{season}_{league}.csv"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < _CACHE_MAX_AGE

    def _hf_parquet_path(self, league: str) -> Optional[Path]:
        if self.hf_config is None:
            return None
        return (
            Path(self.hf_config.local_dir)
            / self.hf_config.dataset_subfolder
            / f"{league}.parquet"
        )

    def _apply_column_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        available_cols = [c for c in self.config.columns_to_keep if c in df.columns]
        df = df[available_cols + [c for c in ("League", "Season") if c in df.columns]]
        return df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])

    # ------------------------------------------------------------------
    # HF Parquet source
    # ------------------------------------------------------------------

    def _load_from_hf_parquet(self, league: str, season: str) -> pd.DataFrame:
        """Load a single season from the HF-downloaded per-league Parquet file."""
        parquet_path = self._hf_parquet_path(league)
        if parquet_path is None or not parquet_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(parquet_path)
            df = df[df["Season"] == season]
            if df.empty:
                return pd.DataFrame()
            return self._apply_column_filter(df)
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_season(self, league: str, season: str) -> pd.DataFrame:
        """Load data for one season/league. Priority: HF Parquet → local cache → web."""

        # 1. HuggingFace Parquet (primary on Render, where no persistent disk exists)
        df = self._load_from_hf_parquet(league, season)
        if not df.empty:
            return df

        # 2. Local 24 h disk cache
        cached = self._cached_path(league, season)
        if self._is_cache_valid(cached):
            try:
                df = pd.read_csv(cached, encoding="utf-8")
                df = self._apply_column_filter(df)
                df["League"] = self.config.leagues.get(league, league)
                df["Season"] = season
                return df
            except Exception:
                pass

        # 3. Download from football-data.co.uk and save to local cache
        url = f"{self.config.base_url}/{season}/{league}.csv"
        try:
            df = pd.read_csv(url, encoding="utf-8", on_bad_lines="skip")
            df.to_csv(cached, index=False)
            df = self._apply_column_filter(df)
            df["League"] = self.config.leagues.get(league, league)
            df["Season"] = season
            return df
        except Exception as e:
            print(f"Error loading {league}/{season}: {e}")
            return pd.DataFrame()

    def load_all(self) -> pd.DataFrame:
        """Load all data for configured leagues and seasons."""
        frames: list[pd.DataFrame] = []
        for league in self.config.leagues:
            for season in self.config.seasons:
                df = self.load_season(league, season)
                if not df.empty:
                    frames.append(df)
                    league_name = self.config.leagues.get(league, league)
                    print(f"  Loaded {league_name}, season {season}: {len(df)} matches")
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        print(f"\nTotal loaded: {len(result)} matches")
        return result

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """Load data from a local CSV file."""
        df = pd.read_csv(path, encoding="utf-8")
        available_cols = [c for c in self.config.columns_to_keep if c in df.columns]
        return df[available_cols].dropna(subset=["HomeTeam", "AwayTeam", "FTR"])

    def save_as_parquet(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Convert a combined DataFrame to per-league Parquet files.

        Each file is named {league_code}.parquet and contains all seasons
        for that league. This is the format expected by _load_from_hf_parquet().
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if "League" not in df.columns:
            raise ValueError("DataFrame must have a 'League' column to split by league")

        league_name_to_code = {v: k for k, v in self.config.leagues.items()}

        for league_name, league_df in df.groupby("League"):
            league_code = league_name_to_code.get(str(league_name), str(league_name))
            parquet_path = output_dir / f"{league_code}.parquet"
            league_df.to_parquet(parquet_path, index=False)
            print(f"  Saved {league_code}.parquet ({len(league_df)} rows)")
