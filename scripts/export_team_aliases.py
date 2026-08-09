"""
Export admin-approved team aliases from Supabase into the committed seed file.

Approvals are made in the admin panel (or `scripts/review_team_names.py`) and
land in the Supabase `team_aliases` table. Training cannot read that table:
CI is deliberately given no Supabase credentials, because the network has no
business in the training path. Anything living only in Supabase is therefore
invisible to the scheduled retrain — which is how a green run silently shipped
an uncalibrated model.

This script closes the gap. Supabase stays the review surface; the seed becomes
the record. Run it after approving names, and commit the result.

Usage:
    uv run python scripts/export_team_aliases.py
    uv run python scripts/export_team_aliases.py --seed-path config/team_aliases.yaml
    uv run python scripts/export_team_aliases.py --dry-run

Environment variables (read from .env):
    SUPABASE_URL          — Supabase project URL
    SUPABASE_SERVICE_KEY  — service_role secret
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from config.config_loader import load_config  # noqa: E402
from src.teams.alias_seed import AliasSeedWriter  # noqa: E402
from src.teams.resolver import StaticTeamAliasRepository  # noqa: E402

#: Credentials the Supabase source needs. Named so a missing one is reported
#: rather than surfacing later as an empty result set.
REQUIRED_ENV_VARS = ("SUPABASE_URL", "SUPABASE_SERVICE_KEY")


def build_source() -> Any:
    """The approved-alias repository, or a raised error explaining why not.

    Raises rather than returning ``None`` because every caller here wants the
    same thing: stop. This script is a deliberate admin action, and a silent
    no-op is exactly the outcome that let the uncalibrated retrain look fine.
    """
    try:
        from src.backend.core.supabase_client import get_supabase_client
        from src.backend.repositories.team_alias_repository import (
            SupabaseTeamAliasRepository,
        )
        from src.backend.services.team_alias_service import TeamAliasService
    except ImportError as exc:  # pragma: no cover - install-time failure
        raise RuntimeError(f"Supabase support is not installed ({exc})") from exc

    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"{' and '.join(missing)} not set (checked .env)")

    try:
        client = get_supabase_client()
    except Exception as exc:
        raise RuntimeError(f"could not connect to Supabase: {exc}") from exc
    return SupabaseTeamAliasRepository(TeamAliasService(client))


def export_aliases(source: Any, seed_path: str | Path) -> int:
    """Write every approved alias into the seed and verify it landed.

    ``SupabaseTeamAliasRepository.get_aliases`` swallows failures by design —
    right for the fixture pipeline, which meets unknown names every run and
    must not die because the database is down. Here it is wrong, so an empty
    result is treated as failure rather than as "nothing to do": the two are
    indistinguishable from the outside, and only one of them is safe.
    """
    approved = list(source.get_aliases())
    if not approved:
        raise RuntimeError(
            "no approved aliases were read from the source — either the table "
            "is empty or the read failed. Nothing was written."
        )

    report = AliasSeedWriter(seed_path).merge(approved)

    # Read the file back through the same repository training uses, rather
    # than trusting the write. A seed that does not parse is worth catching
    # here, not on the next retrain.
    written = set(StaticTeamAliasRepository(seed_path).get_aliases())
    missing = [alias for alias in approved if alias not in written]
    if missing:
        raise RuntimeError(
            f"{len(missing)} aliases did not survive the write, e.g. {missing[0]}"
        )

    print(
        f"Exported {len(approved)} approved aliases to {seed_path}\n"
        f"  {report.added} added, {report.updated} updated, "
        f"{report.unchanged} unchanged\n"
        f"  {report.total} entries in the seed in total"
    )
    print("\nCommit the seed — training reads it, and CI cannot reach Supabase.")
    return len(approved)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export approved team aliases from Supabase into the seed file"
    )
    parser.add_argument(
        "--seed-path",
        default="",
        help="Override the seed path from config (teams.aliases.seed_path)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be exported without writing",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seed_path = args.seed_path or load_config().teams.aliases.seed_path

    try:
        source = build_source()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if args.dry_run:
        approved = list(source.get_aliases())
        print(f"[dry-run] {len(approved)} approved aliases would go to {seed_path}")
        for alias in sorted(approved, key=lambda a: (a.league_code, a.raw_name)):
            print(
                f"  {alias.league_code:10s} {alias.raw_name} → {alias.canonical_name}"
            )
        return

    try:
        export_aliases(source, seed_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
