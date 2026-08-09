"""
Reconcile openfootball team names with the canonical registry.

openfootball names clubs in full ("Sport Lisboa e Benfica") while every model
and dataset here is keyed by football-data's short form ("Benfica"). This
script sorts every distinct spelling into three outcomes:

* **matched outright** — nothing to do; the resolver recognises these natively,
  so writing them to the alias seed would only add noise;
* **needs review** — the club's country has leagues in this project, so there
  is something real to match against. Queued for the Team Aliases panel;
* **untracked country** — no league here at all, so there is nothing to match
  and no decision to make. Reported, never queued.

Nothing is ever matched automatically on similarity. Candidates are narrowed
to the club's own country first, which is what stops "AC Sparta Praha" being
offered "Sparta Rotterdam".

Usage:
    uv run python scripts/queue_european_team_names.py
    uv run python scripts/queue_european_team_names.py --with-qualifiers
    uv run python scripts/queue_european_team_names.py --push
    uv run python scripts/queue_european_team_names.py --limit 20
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

# Supabase credentials live in .env, exactly as the backend and the HF upload
# script expect. Without this the queue silently reports "not configured".
load_dotenv()

from config.config_loader import Config, load_config  # noqa: E402
from src.corpus.supplementary import StaticFileCorpusSource  # noqa: E402
from src.teams.european_names import (  # noqa: E402
    EuropeanNameResolver,
    NameReviewSummary,
)
from src.teams.registry import load_team_registry  # noqa: E402
from src.teams.resolver import (  # noqa: E402
    StaticTeamAliasRepository,
    TeamNameResolver,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile openfootball team names with the canonical registry"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--with-qualifiers",
        action="store_true",
        help="Use the corpus that includes qualifying rounds",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Queue reviewable names in Supabase for the admin panel",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="How many example names to print per bucket",
    )
    return parser.parse_args()


def build_resolver(config: Config) -> EuropeanNameResolver:
    """Wire a resolver from the shipped registry and reviewed alias seed."""
    registry = load_team_registry(config.teams.historical_registry_path)
    names = TeamNameResolver(
        registry,
        StaticTeamAliasRepository(config.teams.aliases.seed_path),
        config.teams.aliases,
    )
    return EuropeanNameResolver(config.european, names)


def corpus_path(config: Config, with_qualifiers: bool) -> str:
    """Which cache to read: main draws, or main draws plus qualifiers."""
    return (
        config.european.qualifiers_cache_path
        if with_qualifiers
        else config.european.cache_path
    )


def queued_names(service: Any) -> set[str]:
    """Names currently in the review queue, or an empty set if unreadable."""
    try:
        return {str(row.get("raw_name", "")) for row in service.list_pending()}
    except Exception:
        return set()


def build_sink() -> tuple[Optional[Any], Optional[Any], str]:
    """The review queue sink, the service behind it, and why it is missing.

    The reason is returned rather than swallowed: "not configured" is
    indistinguishable from a typo'd URL, an absent table or a missing
    dependency, and guessing which costs more than reporting it. The service
    comes back too, so the caller can read the queue afterwards and confirm
    the writes actually landed.
    """
    try:
        from src.backend.core.supabase_client import get_supabase_client
        from src.backend.repositories.team_alias_repository import (
            SupabaseUnresolvedNameSink,
        )
        from src.backend.services.team_alias_service import TeamAliasService
    except ImportError as exc:
        return None, None, f"Supabase support is not installed ({exc})"

    missing = [
        name
        for name in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        return None, None, f"{' and '.join(missing)} not set (checked .env)"

    try:
        client = get_supabase_client()
    except Exception as exc:
        return None, None, f"could not connect to Supabase: {exc}"
    service = TeamAliasService(client)
    return SupabaseUnresolvedNameSink(service), service, ""


def _diagnose(service: Any) -> str:
    """Why nothing landed. Reproduces one write with the errors left in.

    Both ``record_pending`` and the sink swallow failures by design — the
    fixture pipeline meets the same unknown name every run and must not die
    because the queue is down. That is right for the pipeline and wrong here:
    a deliberate admin action reporting success while writing nothing is the
    worst of both.
    """
    try:
        service.list_pending()
    except Exception as exc:
        return f"the review queue is unreadable: {exc}"
    return (
        "the write was rejected. If the team_aliases table does not exist "
        "yet, create it with the SQL in README.md."
    )


def _report(summary: NameReviewSummary, limit: int) -> None:
    print(f"\nDistinct openfootball spellings: {summary.total}")
    print(f"  matched outright   : {len(summary.resolved)}")
    print(f"  needs your review  : {len(summary.reviewable)}")
    print(f"  untracked country  : {len(summary.untrackable)}")

    if summary.reviewable:
        print(f"\nAwaiting review (first {limit}):")
        for resolution in summary.reviewable[:limit]:
            options = ", ".join(resolution.suggestions[:3]) or "(no suggestion)"
            print(f"  {resolution.raw_name:34s} -> {options}")

    if summary.untrackable:
        print(f"\nNo tracked league for these countries (first {limit}):")
        for resolution in summary.untrackable[:limit]:
            print(f"  {resolution.raw_name}")


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    path = corpus_path(config, args.with_qualifiers)
    corpus = StaticFileCorpusSource(path).load()
    if corpus.empty:
        print(
            f"ERROR: no corpus at {path}. "
            "Run scripts/build_european_corpus.py first."
        )
        sys.exit(1)
    print(f"Read {len(corpus)} European matches from {path}")

    summary = build_resolver(config).review(corpus)
    _report(summary, args.limit)

    if not args.push:
        print("\nNothing queued (pass --push to fill the admin review panel)")
        return

    sink, service, reason = build_sink()
    if sink is None or service is None:
        print(f"\nNothing queued — {reason}")
        sys.exit(1)

    before = queued_names(service)
    for resolution in summary.reviewable:
        sink.record(resolution)

    # Read back rather than trust: every layer below here swallows failures,
    # so "queued 101" would otherwise be printed even when nothing was written.
    landed = queued_names(service) - before
    if not landed:
        print(f"\nNothing was queued — {_diagnose(service)}")
        sys.exit(1)
    print(f"\nQueued {len(landed)} names for review in the admin panel")
    if len(landed) < len(summary.reviewable):
        already = len(summary.reviewable) - len(landed)
        print(f"  ({already} were already in the queue)")


if __name__ == "__main__":
    main()
