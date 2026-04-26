"""
Prediction service — reads pre-computed predictions from Supabase.

The ML inference pipeline now runs offline (in GitHub Actions) and uploads
results to the ``predictions`` Supabase table.  This service simply reads
those rows, keeping the backend lightweight and well within memory limits.

An in-memory cache (10 min TTL) avoids hitting Supabase on every request.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config.config_loader import Config
from src.backend.core.supabase_client import get_supabase_client

# Cache TTL in seconds (10 minutes)
CACHE_TTL = 10 * 60
# Maximum number of league:date entries kept in memory
MAX_CACHE_ENTRIES = 24


@dataclass
class LeaguePredictions:
    """Pre-computed predictions for a single league, fetched from Supabase."""

    league_code: str
    league_name: str
    matches: list[dict[str, Any]]
    timestamp: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > CACHE_TTL


class PredictionService:
    """Reads pre-computed predictions from Supabase with in-memory caching."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._cache: OrderedDict[str, LeaguePredictions] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_league_predictions(
        self, league_code: str, target_date: str | None = None
    ) -> LeaguePredictions:
        """Get predictions for a league from Supabase (with in-memory cache)."""
        resolved_date = target_date or datetime.now().strftime("%d/%m/%Y")
        cache_key = f"{league_code}:{resolved_date}"

        # In-memory cache hit
        if cache_key in self._cache and not self._cache[cache_key].is_expired():
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # Fetch from Supabase
        result = self._fetch_from_supabase(league_code, resolved_date)
        self._cache[cache_key] = result
        while len(self._cache) > MAX_CACHE_ENTRIES:
            self._cache.popitem(last=False)
        return result

    def get_all_leagues_predictions(
        self, target_date: str | None = None
    ) -> list[LeaguePredictions]:
        """Get predictions for all leagues that have data for the given date."""
        resolved_date = target_date or datetime.now().strftime("%d/%m/%Y")

        # Try batch fetch from Supabase (single query for all leagues on date)
        results = self._fetch_all_from_supabase(resolved_date)
        return [r for r in results if r.matches]

    def get_available_dates(self) -> list[str]:
        """Return sorted distinct dates that have predictions in Supabase."""
        try:
            sb = get_supabase_client()
            resp = sb.table("predictions").select("match_date").execute()
            rows: list[dict[str, Any]] = resp.data or []  # type: ignore[assignment]
            dates: set[str] = {row["match_date"] for row in rows}
            return sorted(dates, key=lambda d: datetime.strptime(d, "%d/%m/%Y"))
        except Exception as e:
            print(f"Error fetching dates from Supabase: {e}")
            return []

    def invalidate_cache(self, league_code: str | None = None) -> None:
        """Clear the in-memory cache."""
        if league_code:
            keys_to_remove = [k for k in self._cache if k.startswith(league_code)]
            for k in keys_to_remove:
                del self._cache[k]
        else:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Supabase fetchers
    # ------------------------------------------------------------------

    def _fetch_from_supabase(
        self, league_code: str, match_date: str
    ) -> LeaguePredictions:
        """Fetch a single league's predictions from Supabase."""
        league_name = self.config.data.leagues.get(league_code, league_code)
        try:
            sb = get_supabase_client()
            resp = (
                sb.table("predictions")
                .select("payload")
                .eq("league_code", league_code)
                .eq("match_date", match_date)
                .limit(1)
                .execute()
            )
            rows: list[dict[str, Any]] = resp.data or []  # type: ignore[assignment]
            if rows:
                payload: dict[str, Any] = rows[0]["payload"]
                return LeaguePredictions(
                    league_code=str(payload.get("league_code", league_code)),
                    league_name=str(payload.get("league_name", league_name)),
                    matches=payload.get("matches", []),
                )
        except Exception as e:
            print(f"Supabase fetch failed for {league_code}/{match_date}: {e}")

        return LeaguePredictions(
            league_code=league_code,
            league_name=league_name,
            matches=[],
        )

    def _fetch_all_from_supabase(
        self, match_date: str
    ) -> list[LeaguePredictions]:
        """Fetch all leagues' predictions for a date in a single query."""
        results: list[LeaguePredictions] = []
        try:
            sb = get_supabase_client()
            resp = (
                sb.table("predictions")
                .select("league_code, payload")
                .eq("match_date", match_date)
                .execute()
            )
            rows: list[dict[str, Any]] = resp.data or []  # type: ignore[assignment]
            for row in rows:
                payload: dict[str, Any] = row["payload"]
                lc: str = row["league_code"]
                ln = self.config.data.leagues.get(lc, lc)
                lp = LeaguePredictions(
                    league_code=str(payload.get("league_code", lc)),
                    league_name=str(payload.get("league_name", ln)),
                    matches=payload.get("matches", []),
                )
                # Also populate the in-memory cache
                cache_key = f"{lc}:{match_date}"
                self._cache[cache_key] = lp
                results.append(lp)
        except Exception as e:
            print(f"Supabase batch fetch failed for {match_date}: {e}")

        return results
