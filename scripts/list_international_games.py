"""
List upcoming national-team fixtures from FlashScore.

Usage:
    uv run python scripts/list_international_games.py
    uv run python scripts/list_international_games.py --tournaments WC,EURO
    uv run python scripts/list_international_games.py --date 14/06/2026
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.international_fixtures_fetcher import (
    build_international_fixtures_fetcher,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List upcoming national-team fixtures"
    )
    parser.add_argument(
        "--tournaments",
        default=None,
        help="Comma-separated tournament codes (default: all configured)",
    )
    parser.add_argument(
        "--date", default=None, help="Filter by date (DD/MM/YYYY)"
    )
    args = parser.parse_args()

    tournaments = (
        [t.strip() for t in args.tournaments.split(",")]
        if args.tournaments
        else None
    )

    fetcher = build_international_fixtures_fetcher()
    fixtures = fetcher.fetch_upcoming(tournaments=tournaments, target_date=args.date)

    if not fixtures:
        print("No upcoming national-team fixtures found.")
        return

    print(f"\n=== {len(fixtures)} upcoming national-team fixtures ===")
    for fx in fixtures:
        print(
            f"  [{fx.division}] {fx.date} {fx.time}  "
            f"{fx.home_team} vs {fx.away_team}"
        )


if __name__ == "__main__":
    main()
