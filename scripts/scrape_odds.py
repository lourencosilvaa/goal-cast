"""
Fetch today's fixtures and Bet365 odds from football-data.co.uk.

Usage:
    uv run python scripts/scrape_odds.py
    uv run python scripts/scrape_odds.py --league D1
    uv run python scripts/scrape_odds.py --league E0,D1,I1
    uv run python scripts/scrape_odds.py --date 18/04/2026
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from src.scrapers.fixtures_fetcher import DIVISION_MAP, fetch_fixtures


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch today's fixtures and odds")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--league", default=None,
        help="Division code(s), comma-separated (e.g. E0,D1,I1). "
             "Use --list-leagues to see codes.",
    )
    parser.add_argument(
        "--date", default=None,
        help="Date in DD/MM/YYYY format. Defaults to today.",
    )
    parser.add_argument(
        "--list-leagues", action="store_true",
        help="List supported league codes and exit.",
    )
    parser.add_argument(
        "--output", default=None, help="Output file for results"
    )
    args = parser.parse_args()

    if args.list_leagues:
        print("\nSupported league codes:")
        for code, name in sorted(DIVISION_MAP.items()):
            print(f"  {code:4s} - {name}")
        return

    config = load_config(args.config)

    leagues = None
    if args.league:
        leagues = [l.strip() for l in args.league.split(",")]

    print("\n=== Fetching Fixtures ===")
    print(f"Date: {args.date or 'today'}")
    if leagues:
        print(f"Leagues: {', '.join(DIVISION_MAP.get(l, l) for l in leagues)}")
    else:
        print("Leagues: all")

    fixtures = fetch_fixtures(target_date=args.date, leagues=leagues)

    print(f"\nTotal matches found: {len(fixtures)}")

    # Group by league
    by_league: dict[str, list] = {}
    for f in fixtures:
        by_league.setdefault(f.league, []).append(f)

    for league, matches in sorted(by_league.items()):
        print(f"\n  {league} ({len(matches)} matches)")
        for m in sorted(matches, key=lambda x: x.time):
            probs = m.implied_probabilities()
            print(
                f"    {m.time:5s}  {m.home_team:20s} vs {m.away_team:20s}  "
                f"B365: {m.b365_home:.2f} / {m.b365_draw:.2f} / {m.b365_away:.2f}  "
                f"({probs['home']:.0%} / {probs['draw']:.0%} / {probs['away']:.0%})"
            )

    # Save results
    output_path = args.output or f"{config.output.reports_dir}/odds_latest.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            [fx.to_dict() for fx in fixtures],
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    print(f"\nFixtures saved to {output_path}")


if __name__ == "__main__":
    main()
