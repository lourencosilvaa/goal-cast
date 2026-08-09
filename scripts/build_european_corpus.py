"""
Build the European club-competition corpus from openfootball.

Syncs the public-domain openfootball checkout, parses every configured
competition-season into goals-only rows, and writes them to the CSV cache the
training pipeline reads. Run this before a retrain whenever new European
rounds have been played — training itself never reaches for the network, so a
GitHub outage can never silently produce a model without cross-league links.

Two corpora are written: the main draws, and the same plus qualifying rounds.
Which one to train on is decided by measurement, not assumption.

Usage:
    uv run python scripts/build_european_corpus.py
    uv run python scripts/build_european_corpus.py --no-sync
    uv run python scripts/build_european_corpus.py --seasons 2526,2425
    uv run python scripts/build_european_corpus.py --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from config.config_loader import EuropeanConfig, load_config  # noqa: E402
from src.corpus.openfootball.repository import OpenFootballRepository  # noqa: E402
from src.corpus.openfootball.source import OpenFootballCorpusSource  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the European corpus from openfootball"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config file providing the competition map, seasons and cache paths",
    )
    parser.add_argument(
        "--seasons",
        default=None,
        help="Comma-separated season codes, e.g. 2526,2425 (default: all configured)",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Parse the existing checkout without contacting GitHub",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without writing it",
    )
    return parser.parse_args()


def _narrow(config: EuropeanConfig, seasons: str | None) -> EuropeanConfig:
    """A copy of the config restricted to the requested seasons.

    Copied rather than mutated so a partial run can never leave the loaded
    configuration looking like a full one.
    """
    if not seasons:
        return config
    wanted = [season.strip() for season in seasons.split(",")]
    narrowed = config.model_copy(deep=True)
    narrowed.seasons = [s for s in narrowed.seasons if s in wanted]
    return narrowed


def _report(frame: pd.DataFrame, label: str) -> None:
    print(f"\n{label}: {len(frame)} matches")
    if frame.empty:
        return
    for competition, rows in frame.groupby("Div"):
        span = f"{rows['Date'].min():%Y-%m-%d} to {rows['Date'].max():%Y-%m-%d}"
        seasons = rows["Season"].nunique()
        print(
            f"  {competition:6s} {len(rows):5d} matches  {seasons:2d} seasons  {span}"
        )


def _write(frame: pd.DataFrame, path_text: str) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"  wrote {len(frame)} matches to {path}")


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    european = _narrow(config.european, args.seasons)

    if not european.enabled:
        print("European track is disabled in config — nothing built")
        return

    repository = OpenFootballRepository(european)
    if args.no_sync:
        print(f"Skipping sync; parsing existing checkout at {repository.path}")
    else:
        print(f"Syncing {european.repo_url} -> {repository.path}")
        if not repository.sync():
            print("WARNING: sync failed — parsing whatever is already on disk")

    if not repository.path.is_dir():
        print(f"ERROR: no checkout at {repository.path}. Nothing to parse.")
        sys.exit(1)

    main_draws = OpenFootballCorpusSource(european, repository).load()
    with_qualifiers = OpenFootballCorpusSource(
        european, repository, include_qualifiers=True
    ).load()

    _report(main_draws, "Main draws")
    _report(with_qualifiers, "Main draws + qualifiers")

    if main_draws.empty:
        print("\nWARNING: no matches parsed — nothing written")
        return

    if args.dry_run:
        print("\nDry run — nothing written")
        return

    print()
    _write(main_draws, european.cache_path)
    _write(with_qualifiers, european.qualifiers_cache_path)


if __name__ == "__main__":
    main()
