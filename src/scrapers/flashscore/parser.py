from datetime import datetime, timezone
from typing import Optional

from src.scrapers.base_scraper import FlashScoreFixture

_FIELD_SEP = "\xf7"
_ENTRY_SEP = "~"
_RECORD_SEP = "\xac"


class FlashScoreParser:
    """Parses FlashScore's pipe-delimited feed format into FlashScoreFixture objects."""

    def parse_fixtures(
        self, raw: str, *, league: str, country: str
    ) -> list[FlashScoreFixture]:
        if not raw:
            return []

        fixtures: list[FlashScoreFixture] = []
        for entry in raw.split(_ENTRY_SEP):
            entry = entry.strip()
            if not entry or not entry.startswith("ZEE"):
                continue
            fixture = self._parse_entry(entry, league=league, country=country)
            if fixture is not None:
                fixtures.append(fixture)
        return fixtures

    def _parse_entry(
        self, entry: str, *, league: str, country: str
    ) -> Optional[FlashScoreFixture]:
        fields: dict[str, str] = {}
        for record in entry.split(_RECORD_SEP):
            record = record.strip()
            if _FIELD_SEP not in record:
                continue
            key, _, value = record.partition(_FIELD_SEP)
            fields[key.strip()] = value.strip()

        match_id = fields.get("ZEE", "")
        home_team = fields.get("AA", "")
        away_team = fields.get("AB", "")
        timestamp_raw = fields.get("AD", "")
        status_raw = fields.get("AE", "scheduled")
        home_score_raw = fields.get("AF", "")
        away_score_raw = fields.get("AG", "")

        if not match_id or not home_team or not away_team:
            return None

        match_datetime = self._parse_timestamp(timestamp_raw)
        home_score = self._parse_score(home_score_raw)
        away_score = self._parse_score(away_score_raw)

        return FlashScoreFixture(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            country=country,
            match_datetime=match_datetime,
            status=status_raw or "scheduled",
            home_score=home_score,
            away_score=away_score,
            source_url=f"https://www.flashscore.com/match/{match_id}/",
        )

    @staticmethod
    def _parse_timestamp(raw: str) -> str:
        if not raw:
            return ""
        try:
            ts = int(raw)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except (ValueError, OSError):
            return raw

    @staticmethod
    def _parse_score(raw: str) -> Optional[int]:
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None
