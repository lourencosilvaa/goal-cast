"""
Fetch today's fixtures and odds from football-data.co.uk fixtures CSV.

Replaces the Playwright-based betting site scrapers with a simple CSV download.
The CSV includes Bet365 odds which serve as the bookmaker baseline.

Downloads the fixtures CSV once and caches it locally for 1 hour.
"""

import csv
import io
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from src.scrapers.base_scraper import BaseFixtureScraper

FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"
_CACHE_DIR = Path("datasets/cache")
_FIXTURES_CACHE = _CACHE_DIR / "fixtures.csv"
_FIXTURES_CACHE_AGE = 60 * 60  # 1 hour


FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"

# Map CSV division codes to league names
DIVISION_MAP = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "P1": "Liga Portugal",
}


@dataclass
class Fixture:
    """A scheduled match with bookmaker odds from the fixtures CSV."""

    division: str
    league: str
    date: str
    time: str
    home_team: str
    away_team: str
    b365_home: float
    b365_draw: float
    b365_away: float

    def implied_probabilities(self) -> dict[str, float]:
        """Convert B365 odds to normalized implied probabilities."""
        raw_h = 1 / self.b365_home if self.b365_home > 0 else 0
        raw_d = 1 / self.b365_draw if self.b365_draw > 0 else 0
        raw_a = 1 / self.b365_away if self.b365_away > 0 else 0
        total = raw_h + raw_d + raw_a
        if total == 0:
            return {"home": 0, "draw": 0, "away": 0}
        return {
            "home": raw_h / total,
            "draw": raw_d / total,
            "away": raw_a / total,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "division": self.division,
            "league": self.league,
            "date": self.date,
            "time": self.time,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "b365_odds": {
                "home": self.b365_home,
                "draw": self.b365_draw,
                "away": self.b365_away,
            },
            "implied_probabilities": self.implied_probabilities(),
        }


def _get_fixtures_csv_text() -> str:
    """Get fixtures CSV text, using local cache if fresh enough."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if _FIXTURES_CACHE.exists():
        age = time.time() - _FIXTURES_CACHE.stat().st_mtime
        if age < _FIXTURES_CACHE_AGE:
            return _FIXTURES_CACHE.read_text(encoding="utf-8-sig")

    resp = requests.get(FIXTURES_CSV_URL, timeout=30)
    resp.raise_for_status()
    text = resp.content.decode("utf-8-sig")
    _FIXTURES_CACHE.write_text(text, encoding="utf-8")
    return text


def fetch_fixtures(
    target_date: str | None = None,
    leagues: list[str] | None = None,
) -> list[Fixture]:
    """
    Download fixtures CSV and return parsed fixtures.

    Args:
        target_date: Date string in DD/MM/YYYY format. Defaults to today.
        leagues: List of division codes (e.g. ["E0", "D1"]) to filter.
                 None means all supported leagues.
    """
    if target_date is None:
        target_date = datetime.now().strftime("%d/%m/%Y")

    text = _get_fixtures_csv_text()

    reader = csv.DictReader(io.StringIO(text))
    fixtures: list[Fixture] = []

    for row in reader:
        row_date = row.get("Date", "")
        div = row.get("Div", "")

        if row_date != target_date:
            continue

        if leagues and div not in leagues:
            continue

        # Only include leagues we support (or all if no filter)
        league_name = DIVISION_MAP.get(div, div)

        try:
            b365_h = float(row.get("B365H", 0) or 0)
            b365_d = float(row.get("B365D", 0) or 0)
            b365_a = float(row.get("B365A", 0) or 0)
        except (ValueError, TypeError):
            b365_h = b365_d = b365_a = 0.0

        fixtures.append(
            Fixture(
                division=div,
                league=league_name,
                date=row_date,
                time=row.get("Time", ""),
                home_team=row.get("HomeTeam", ""),
                away_team=row.get("AwayTeam", ""),
                b365_home=b365_h,
                b365_draw=b365_d,
                b365_away=b365_a,
            )
        )

    # Fallback 1: if CSV had no fixtures for this date, try The Odds API
    if not fixtures:
        fixtures = _fetch_odds_api_fixtures(target_date, leagues)

    # Fallback 2: if OddsAPI also empty, try FlashScore
    if not fixtures:
        fixtures = _fetch_flashscore_fixtures(target_date, leagues)

    return fixtures


def fetch_available_dates(leagues: list[str] | None = None) -> list[str]:
    """Return sorted list of distinct dates (DD/MM/YYYY) from CSV + Odds API.

    Only includes dates for the requested league codes (or all supported
    leagues if *leagues* is ``None``).
    """
    text = _get_fixtures_csv_text()
    reader = csv.DictReader(io.StringIO(text))
    dates: set[str] = set()
    for row in reader:
        div = row.get("Div", "")
        if leagues and div not in leagues:
            continue
        if div not in DIVISION_MAP:
            continue
        d = row.get("Date", "")
        if d:
            dates.add(d)

    # Merge dates from The Odds API (if configured)
    try:
        from src.scrapers.odds_api_fetcher import (
            _get_api_key,
            fetch_available_dates_from_odds_api,
        )

        api_key = _get_api_key()
        if api_key:
            odds_api_dates = fetch_available_dates_from_odds_api(api_key, leagues)
            dates.update(odds_api_dates)
    except Exception:
        pass  # Odds API unavailable — use CSV dates only

    # Sort chronologically
    return sorted(dates, key=lambda d: datetime.strptime(d, "%d/%m/%Y"))


def _fetch_odds_api_fixtures(
    target_date: str,
    leagues: list[str] | None = None,
) -> list[Fixture]:
    """Fetch fixtures from The Odds API as a fallback source."""
    try:
        from src.scrapers.odds_api_fetcher import (
            _get_api_key,
            fetch_fixtures_from_odds_api,
        )

        api_key = _get_api_key()
        if not api_key:
            return []
        return fetch_fixtures_from_odds_api(
            api_key, target_date=target_date, leagues=leagues, include_odds=True
        )
    except Exception as e:
        print(f"[OddsAPI fallback] Error: {e}")
        return []


def _build_flashscore_scraper() -> "BaseFixtureScraper":
    """Instantiate a FlashScoreScraper from project config."""
    from config.config_loader import load_config
    from src.scrapers.flashscore.http_client import FlashScoreHttpClient
    from src.scrapers.flashscore.parser import FlashScoreParser
    from src.scrapers.flashscore.playwright_client import FlashScorePlaywrightClient
    from src.scrapers.flashscore.scraper import FlashScoreScraper

    cfg = load_config()
    fs_cfg = cfg.scrapers.flashscore

    class _MergedConfig:
        base_url = fs_cfg.base_url
        http_enabled = fs_cfg.http_enabled
        playwright_fallback_enabled = fs_cfg.playwright_fallback_enabled
        leagues = fs_cfg.leagues
        user_agent = cfg.scrapers.user_agent
        request_timeout = cfg.scrapers.request_timeout

    merged = _MergedConfig()
    return FlashScoreScraper(
        merged,
        FlashScoreHttpClient(merged),
        FlashScorePlaywrightClient(merged),
        FlashScoreParser(),
    )


def _fetch_flashscore_fixtures(
    target_date: str,
    leagues: list[str] | None = None,
) -> list[Fixture]:
    """Fetch fixtures from FlashScore as a tertiary fallback source."""
    try:
        from src.scrapers.flashscore.exceptions import FlashScoreUnavailableError

        scraper = _build_flashscore_scraper()
        if leagues is None:
            leagues = list(scraper.get_available_leagues().keys())  # type: ignore[union-attr]

        fixtures: list[Fixture] = []
        for league_code in leagues:
            try:
                fs_fixtures = scraper.scrape_fixtures(league_code)  # type: ignore[union-attr]
                for fs in fs_fixtures:
                    league_name = DIVISION_MAP.get(league_code, fs.league)
                    fixtures.append(
                        Fixture(
                            division=league_code,
                            league=league_name,
                            date=target_date,
                            time=(
                                fs.match_datetime[11:16]
                                if len(fs.match_datetime) >= 16
                                else ""
                            ),
                            home_team=fs.home_team,
                            away_team=fs.away_team,
                            b365_home=0.0,
                            b365_draw=0.0,
                            b365_away=0.0,
                        )
                    )
            except FlashScoreUnavailableError:
                continue
        return fixtures
    except Exception as e:
        print(f"[FlashScore fallback] Error: {e}")
        return []


def fetch_fixtures_for_matches(
    matches: list[tuple[str, str]],
    target_date: str | None = None,
) -> list[Fixture]:
    """
    Fetch fixtures and filter to specific matches.

    Args:
        matches: List of (home_team, away_team) tuples.
        target_date: Date string in DD/MM/YYYY format. Defaults to today.
    """
    all_fixtures = fetch_fixtures(target_date=target_date)

    # Normalize match names for fuzzy matching
    def normalize(name: str) -> str:
        return name.lower().strip()

    result: list[Fixture] = []
    for fixture in all_fixtures:
        fh = normalize(fixture.home_team)
        fa = normalize(fixture.away_team)
        for home, away in matches:
            h = normalize(home)
            a = normalize(away)
            if (h in fh or fh in h) and (a in fa or fa in a):
                result.append(fixture)
                break

    return result
