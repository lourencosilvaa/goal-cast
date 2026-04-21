"""
Fetch upcoming fixtures and odds from The Odds API (https://the-odds-api.com).

Provides an alternative/fallback fixture source when football-data.co.uk's
fixtures CSV doesn't yet include upcoming dates.

The free tier offers 500 requests/month.
- GET /v4/sports — free (no quota cost)
- GET /v4/sports/{sport}/events — free (no quota cost)
- GET /v4/sports/{sport}/odds — 1 credit per region per market

Set the API key via config or THE_ODDS_API_KEY environment variable.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from src.scrapers.fixtures_fetcher import DIVISION_MAP, Fixture

_CACHE_DIR = Path("datasets/cache")
_CACHE_AGE = 60 * 60  # 1 hour

# Map our division codes to The Odds API sport keys
DIVISION_TO_SPORT_KEY: dict[str, str] = {
    "E0": "soccer_epl",
    "SP1": "soccer_spain_la_liga",
    "D1": "soccer_germany_bundesliga",
    "I1": "soccer_italy_serie_a",
    "F1": "soccer_france_ligue_one",
    "P1": "soccer_portugal_primeira_liga",
}

BASE_URL = "https://api.the-odds-api.com/v4"


def _get_api_key(config_key: str = "") -> str:
    """Get API key from env var or config, preferring env var."""
    return os.environ.get("THE_ODDS_API_KEY", config_key)


def _cached_json_path(sport_key: str, endpoint: str) -> Path:
    return _CACHE_DIR / f"odds_api_{sport_key}_{endpoint}.json"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < _CACHE_AGE


def fetch_events(
    api_key: str,
    sport_key: str,
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch upcoming events for a sport. This endpoint is FREE (no quota cost).

    Returns list of event dicts with id, sport_key, home_team, away_team,
    commence_time.
    """
    import json

    cache_path = _cached_json_path(sport_key, "events")
    if _is_cache_valid(cache_path):
        return json.loads(cache_path.read_text())

    params: dict[str, str] = {"apiKey": api_key}
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if commence_time_to:
        params["commenceTimeTo"] = commence_time_to

    resp = requests.get(
        f"{BASE_URL}/sports/{sport_key}/events",
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    events = resp.json()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(events))
    return events


def fetch_odds(
    api_key: str,
    sport_key: str,
    regions: str = "eu",
    markets: str = "h2h",
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch upcoming events WITH odds. Costs 1 credit per region per market.

    Returns list of event dicts including bookmakers with h2h odds.
    """
    import json

    cache_path = _cached_json_path(sport_key, "odds")
    if _is_cache_valid(cache_path):
        return json.loads(cache_path.read_text())

    params: dict[str, str] = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    }
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if commence_time_to:
        params["commenceTimeTo"] = commence_time_to

    resp = requests.get(
        f"{BASE_URL}/sports/{sport_key}/odds",
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    events = resp.json()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(events))

    # Log remaining quota from response headers
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"[OddsAPI] Quota: {used} used, {remaining} remaining")

    return events


def _extract_best_odds(event: dict) -> tuple[float, float, float]:
    """Extract best (highest) h2h odds from bookmakers in an event."""
    best_h, best_d, best_a = 0.0, 0.0, 0.0

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                price = outcome.get("price", 0)
                name = outcome.get("name", "")
                if name == event.get("home_team"):
                    best_h = max(best_h, price)
                elif name == event.get("away_team"):
                    best_a = max(best_a, price)
                elif name == "Draw":
                    best_d = max(best_d, price)
    return best_h, best_d, best_a


def _extract_avg_odds(event: dict) -> tuple[float, float, float]:
    """Extract average h2h odds across all bookmakers."""
    odds_h: list[float] = []
    odds_d: list[float] = []
    odds_a: list[float] = []

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                price = outcome.get("price", 0)
                name = outcome.get("name", "")
                if name == event.get("home_team"):
                    odds_h.append(price)
                elif name == event.get("away_team"):
                    odds_a.append(price)
                elif name == "Draw":
                    odds_d.append(price)

    avg_h = sum(odds_h) / len(odds_h) if odds_h else 0.0
    avg_d = sum(odds_d) / len(odds_d) if odds_d else 0.0
    avg_a = sum(odds_a) / len(odds_a) if odds_a else 0.0
    return avg_h, avg_d, avg_a


def events_to_fixtures(
    events: list[dict],
    division: str,
    with_odds: bool = False,
) -> list[Fixture]:
    """Convert Odds API events to our Fixture dataclass format."""
    league_name = DIVISION_MAP.get(division, division)
    fixtures: list[Fixture] = []

    for event in events:
        commence = event.get("commence_time", "")
        try:
            dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue

        date_str = dt.strftime("%d/%m/%Y")
        time_str = dt.strftime("%H:%M")

        if with_odds:
            b365_h, b365_d, b365_a = _extract_avg_odds(event)
        else:
            b365_h = b365_d = b365_a = 0.0

        fixtures.append(
            Fixture(
                division=division,
                league=league_name,
                date=date_str,
                time=time_str,
                home_team=event.get("home_team", ""),
                away_team=event.get("away_team", ""),
                b365_home=round(b365_h, 2),
                b365_draw=round(b365_d, 2),
                b365_away=round(b365_a, 2),
            )
        )

    return fixtures


def fetch_fixtures_from_odds_api(
    api_key: str,
    target_date: str | None = None,
    leagues: list[str] | None = None,
    include_odds: bool = True,
) -> list[Fixture]:
    """
    Fetch fixtures (and optionally odds) from The Odds API for the given date.

    Args:
        api_key: The Odds API key.
        target_date: DD/MM/YYYY format. Defaults to today.
        leagues: Division codes to fetch (e.g. ["E0", "D1"]).
                 None means all supported leagues.
        include_odds: If True, fetch odds too (costs 1 credit per league).
                      If False, use the free events endpoint (no odds).
    """
    if not api_key:
        return []

    if target_date is None:
        target_date = datetime.now().strftime("%d/%m/%Y")

    # Parse target date to create ISO time range for that day
    target_dt = datetime.strptime(target_date, "%d/%m/%Y")
    time_from = target_dt.strftime("%Y-%m-%dT00:00:00Z")
    time_to = target_dt.strftime("%Y-%m-%dT23:59:59Z")

    divisions = leagues or list(DIVISION_TO_SPORT_KEY.keys())
    all_fixtures: list[Fixture] = []

    for div in divisions:
        sport_key = DIVISION_TO_SPORT_KEY.get(div)
        if not sport_key:
            continue

        try:
            if include_odds:
                events = fetch_odds(
                    api_key,
                    sport_key,
                    commence_time_from=time_from,
                    commence_time_to=time_to,
                )
            else:
                events = fetch_events(
                    api_key,
                    sport_key,
                    commence_time_from=time_from,
                    commence_time_to=time_to,
                )

            fixtures = events_to_fixtures(events, div, with_odds=include_odds)
            all_fixtures.extend(fixtures)
        except requests.RequestException as e:
            print(f"[OddsAPI] Error fetching {sport_key}: {e}")
            continue

    return all_fixtures


def fetch_available_dates_from_odds_api(
    api_key: str,
    leagues: list[str] | None = None,
) -> list[str]:
    """
    Get all available fixture dates from The Odds API (free, no quota cost).

    Uses the events endpoint which returns upcoming matches without odds.
    """
    if not api_key:
        return []

    divisions = leagues or list(DIVISION_TO_SPORT_KEY.keys())
    dates: set[str] = set()

    for div in divisions:
        sport_key = DIVISION_TO_SPORT_KEY.get(div)
        if not sport_key:
            continue

        try:
            events = fetch_events(api_key, sport_key)
            for event in events:
                commence = event.get("commence_time", "")
                try:
                    dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    dates.add(dt.strftime("%d/%m/%Y"))
                except (ValueError, AttributeError):
                    continue
        except requests.RequestException as e:
            print(f"[OddsAPI] Error fetching dates for {sport_key}: {e}")
            continue

    return sorted(dates, key=lambda d: datetime.strptime(d, "%d/%m/%Y"))
