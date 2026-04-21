import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel


class ConfigurationError(Exception):
    pass


class OrgLoaderConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.football-data.org/v4"
    request_timeout: int = 10
    competitions: dict[str, str] = {}


_CACHE_MAX_AGE = 24 * 60 * 60  # 24 hours


class FootballDataOrgLoader:
    """
    Loads historical match data from the football-data.org v4 REST API.
    Covers UCL (CL), UEL (EL), UECL (ECL), FA Cup (FA), Copa del Rey (CDR).

    Requires a free API key from https://www.football-data.org/
    """

    def __init__(
        self, config: OrgLoaderConfig, cache_dir: str = "datasets/cache/org"
    ) -> None:
        if not config.api_key:
            raise ConfigurationError(
                "api_key must be set in OrgLoaderConfig — "
                "never hardcode it; use env var override"
            )
        self._config = config
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def supported_competitions(self) -> dict[str, str]:
        return dict(self._config.competitions)

    def _cached_path(self, competition: str, season: str) -> Path:
        return self._cache_dir / f"{competition}_{season}.json"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        return (time.time() - path.stat().st_mtime) < _CACHE_MAX_AGE

    def _fetch_matches(self, competition: str, season: str) -> list[dict]:
        season_param = f"20{season[:2]}" if len(season) == 4 else season
        url = f"{self._config.base_url}/competitions/{competition}/matches"
        headers = {"X-Auth-Token": self._config.api_key}
        params = {"season": season_param}

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=self._config.request_timeout,
        )
        response.raise_for_status()
        result: list[dict] = response.json().get("matches", [])
        return result

    def load_season(self, competition: str, season: str) -> pd.DataFrame:
        """
        Load finished matches for a competition/season.
        Results are cached for 24h to avoid hitting the rate limit.
        """
        cached = self._cached_path(competition, season)

        if self._is_cache_valid(cached):
            raw_matches = pd.read_json(cached).to_dict(orient="records")
        else:
            raw_matches = self._fetch_matches(competition, season)
            pd.DataFrame(raw_matches).to_json(cached, orient="records")

        return self._to_dataframe(raw_matches, competition)

    def fetch_upcoming_fixtures(self, competition: str, target_date: str) -> list[dict]:
        """
        Return SCHEDULED matches for *competition* on *target_date* (DD/MM/YYYY).
        Each dict has keys: home_team, away_team, league, date, time.
        """
        try:
            dt = datetime.strptime(target_date, "%d/%m/%Y")
        except ValueError:
            return []

        target_iso = dt.strftime("%Y-%m-%d")
        url = f"{self._config.base_url}/competitions/{competition}/matches"
        headers = {"X-Auth-Token": self._config.api_key}
        params: dict = {"status": "SCHEDULED"}

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=self._config.request_timeout,
        )
        response.raise_for_status()
        raw: list[dict] = response.json().get("matches", [])

        league_name = self._config.competitions.get(competition, competition)
        fixtures = []
        for m in raw:
            utc_date = m.get("utcDate", "")
            match_date = utc_date[:10]
            if match_date != target_iso:
                continue
            home = m.get("homeTeam", {}).get("name", "")
            away = m.get("awayTeam", {}).get("name", "")
            if not home or not away:
                continue
            time_str = utc_date[11:16] if len(utc_date) >= 16 else ""
            fixtures.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "league": league_name,
                    "date": target_date,
                    "time": time_str,
                    "division": competition,
                }
            )
        return fixtures

    def fetch_available_upcoming_dates(self, competition: str) -> list[str]:
        """Return DD/MM/YYYY dates that have SCHEDULED matches for *competition*."""
        url = f"{self._config.base_url}/competitions/{competition}/matches"
        headers = {"X-Auth-Token": self._config.api_key}
        params: dict = {"status": "SCHEDULED"}

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=self._config.request_timeout,
        )
        response.raise_for_status()
        raw: list[dict] = response.json().get("matches", [])

        dates: set[str] = set()
        for m in raw:
            utc_date = m.get("utcDate", "")[:10]
            if not utc_date:
                continue
            try:
                dt = datetime.strptime(utc_date, "%Y-%m-%d")
                dates.add(dt.strftime("%d/%m/%Y"))
            except ValueError:
                continue
        return sorted(dates, key=lambda d: datetime.strptime(d, "%d/%m/%Y"))

    def _to_dataframe(self, matches: list[dict], competition: str) -> pd.DataFrame:
        rows = []
        league_name = self._config.competitions.get(competition, competition)
        for m in matches:
            if m.get("status") != "FINISHED":
                continue
            score = m.get("score", {})
            ft = score.get("fullTime", {})
            ht = score.get("halfTime", {})
            rows.append(
                {
                    "Date": m.get("utcDate", "")[:10],
                    "HomeTeam": m.get("homeTeam", {}).get("name", ""),
                    "AwayTeam": m.get("awayTeam", {}).get("name", ""),
                    "FTHG": ft.get("home"),
                    "FTAG": ft.get("away"),
                    "HTHG": ht.get("home"),
                    "HTAG": ht.get("away"),
                    "League": league_name,
                    "Season": season if (season := competition) else competition,
                }
            )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["Season"] = competition
        return df
