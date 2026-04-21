import time
from pathlib import Path

import pandas as pd

from config.config_loader import DataConfig

# Local cache directory and max age (24 hours)
_CACHE_DIR = Path("datasets/cache")
_CACHE_MAX_AGE = 24 * 60 * 60  # seconds


class FootballDataLoader:
    """
    Historical football match data loader.
    Source: football-data.co.uk

    Downloads CSVs once and caches them locally under datasets/cache/.
    Re-downloads if the cached file is older than 24 hours.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cached_path(self, league: str, season: str) -> Path:
        return _CACHE_DIR / f"{season}_{league}.csv"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < _CACHE_MAX_AGE

    def load_season(self, league: str, season: str) -> pd.DataFrame:
        """Load data for a single season and league, using local cache."""
        cached = self._cached_path(league, season)

        if self._is_cache_valid(cached):
            try:
                df = pd.read_csv(cached, encoding="utf-8")
                available_cols = [
                    c for c in self.config.columns_to_keep if c in df.columns
                ]
                df = df[available_cols].dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
                df["League"] = self.config.leagues.get(league, league)
                df["Season"] = season
                return df
            except Exception:
                pass  # Fall through to re-download

        # Download from web and save locally
        url = f"{self.config.base_url}/{season}/{league}.csv"
        try:
            df = pd.read_csv(url, encoding="utf-8", on_bad_lines="skip")
            # Save raw CSV to cache
            df.to_csv(cached, index=False)
            available_cols = [c for c in self.config.columns_to_keep if c in df.columns]
            df = df[available_cols].dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
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
                    print(
                        f"  Loaded {league_name}, season {season}: "
                        f"{len(df)} matches"
                    )
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
