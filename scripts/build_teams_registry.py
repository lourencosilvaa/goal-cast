"""
Build the static per-league teams registry served as the offline fallback for
GET /api/predictions/teams.

Reads the latest cached season CSV per league from the raw CSV cache and writes
a ``{league_code: [team, ...]}`` JSON file. Uses the stdlib ``csv`` module only,
so it stays runnable wherever the cache exists — no pandas required.

Run this after a season's fixtures are known (promotions/relegations change the
team lists), then commit the regenerated file:

Usage:
    uv run python scripts/build_teams_registry.py
    uv run python scripts/build_teams_registry.py --cache-dir datasets/cache
    uv run python scripts/build_teams_registry.py --config config/config.yaml
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config  # noqa: E402


@dataclass(frozen=True)
class TeamsRegistrySpec:
    """Everything the builder needs, so no argument list has to be threaded."""

    cache_dir: Path
    output_path: Path
    league_codes: list[str]


class TeamsRegistryBuilder:
    """Extracts team names per league from the newest cached season CSV."""

    ENCODING: ClassVar[str] = "utf-8"
    HOME_COLUMN: ClassVar[str] = "HomeTeam"
    AWAY_COLUMN: ClassVar[str] = "AwayTeam"
    JSON_INDENT: ClassVar[int] = 2

    def __init__(self, spec: TeamsRegistrySpec) -> None:
        self.spec = spec

    def _latest_season_file(self, league: str) -> Path | None:
        """Cache files are named ``{season}_{league}.csv``; seasons sort
        lexicographically (e.g. 2425 < 2526), so the last one is the newest."""
        files = sorted(self.spec.cache_dir.glob(f"*_{league}.csv"))
        return files[-1] if files else None

    def _read_teams(self, path: Path) -> list[str]:
        teams: set[str] = set()
        with path.open(encoding=self.ENCODING, newline="") as handle:
            for row in csv.DictReader(handle):
                for column in (self.HOME_COLUMN, self.AWAY_COLUMN):
                    name = (row.get(column) or "").strip()
                    if name:
                        teams.add(name)
        return sorted(teams)

    def build(self) -> dict[str, list[str]]:
        registry: dict[str, list[str]] = {}
        if not self.spec.cache_dir.is_dir():
            return registry
        for league in self.spec.league_codes:
            path = self._latest_season_file(league)
            if path is None:
                continue
            try:
                teams = self._read_teams(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            if teams:
                registry[league] = teams
        return registry

    def write(self, registry: dict[str, list[str]]) -> None:
        self.spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.spec.output_path.write_text(
            json.dumps(registry, indent=self.JSON_INDENT, ensure_ascii=False) + "\n",
            encoding=self.ENCODING,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the static teams registry")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config file providing the league map and registry output path",
    )
    parser.add_argument(
        "--cache-dir",
        default="datasets/cache",
        help="Directory containing the raw per-season CSV cache",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Override the registry output path from config",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    spec = TeamsRegistrySpec(
        cache_dir=Path(args.cache_dir),
        output_path=Path(args.output or config.teams.registry_path),
        league_codes=list(config.data.leagues),
    )
    builder = TeamsRegistryBuilder(spec)
    registry = builder.build()
    if not registry:
        print(f"WARNING: no team data found in {spec.cache_dir} — nothing written")
        return
    builder.write(registry)
    for league, teams in registry.items():
        print(f"  {league}: {len(teams)} teams")
    print(f"Wrote {spec.output_path}")


if __name__ == "__main__":
    main()
