"""Project approved aliases into the committed seed file.

Approvals are made in Supabase, but training must not depend on Supabase. CI is
given no credentials on purpose — the network has no business in the training
path — so anything living only in that table is invisible to the scheduled
retrain. That is not hypothetical: it is why the retrain silently produced an
uncalibrated model while reporting success.

This writer resolves the split by making Supabase the *review* surface and the
seed the *record*. After an export, every input training needs is on disk and
in git, where it is also reviewable in a diff rather than invisible in a table.

The merge is one-way and additive. Entries already in the seed that an export
does not mention are left untouched, so a hand-written FlashScore alias
survives an export of European approvals.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable

import yaml

from src.teams.resolver import TeamAlias


@dataclass(frozen=True)
class SeedMergeReport:
    """What a merge changed, for the caller to print and act on."""

    added: int
    updated: int
    unchanged: int
    total: int


class AliasSeedWriter:
    """Merges :class:`TeamAlias` records into the seed YAML file.

    Accepts any iterable of aliases — the same type
    ``TeamAliasRepository.get_aliases()`` returns — so the source stays
    interchangeable and the writer is testable without a database.
    """

    ENCODING: ClassVar[str] = "utf-8"
    ALIASES_KEY: ClassVar[str] = "aliases"
    COMMENT_PREFIX: ClassVar[str] = "#"

    def __init__(self, seed_path: str | Path) -> None:
        self._seed_path = Path(seed_path)

    def merge(self, aliases: Iterable[TeamAlias]) -> SeedMergeReport:
        """Write ``aliases`` into the seed, preserving everything else."""
        by_scope = self._read()
        added = updated = unchanged = 0

        for alias in aliases:
            scope = alias.league_code.strip()
            raw = alias.raw_name.strip()
            canonical = alias.canonical_name.strip()
            if not scope or not raw or not canonical:
                continue
            entries = by_scope.setdefault(scope, {})
            existing = entries.get(raw)
            if existing is None:
                added += 1
            elif existing != canonical:
                updated += 1
            else:
                unchanged += 1
            entries[raw] = canonical

        self._write(by_scope)
        total = sum(len(entries) for entries in by_scope.values())
        return SeedMergeReport(
            added=added, updated=updated, unchanged=unchanged, total=total
        )

    def _read(self) -> dict[str, dict[str, str]]:
        """Existing entries, or empty for a seed that does not exist yet.

        A seed that exists but cannot be parsed raises instead: overwriting it
        would destroy the very record this file is meant to be.
        """
        try:
            text = self._seed_path.read_text(encoding=self.ENCODING)
        except OSError:
            return {}
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"{self._seed_path} is not valid YAML: {exc}") from exc
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"{self._seed_path} does not contain a mapping")
        by_scope = raw.get(self.ALIASES_KEY)
        if by_scope is None:
            return {}
        if not isinstance(by_scope, dict):
            raise ValueError(
                f"{self._seed_path}: '{self.ALIASES_KEY}' must be a mapping of "
                f"scope -> {{scraped: canonical}}"
            )
        result: dict[str, dict[str, str]] = {}
        for scope, entries in by_scope.items():
            if not isinstance(entries, dict):
                raise ValueError(f"{self._seed_path}: scope '{scope}' is not a mapping")
            result[str(scope)] = {str(k): str(v) for k, v in entries.items()}
        return result

    def _header(self) -> str:
        """The leading comment block, which ``yaml.dump`` would otherwise drop.

        It documents the rule that every mapping is human-validated — worth
        more in this file than in most, since the file is the audit trail.
        """
        try:
            text = self._seed_path.read_text(encoding=self.ENCODING)
        except OSError:
            return ""
        lines: list[str] = []
        for line in text.splitlines(keepends=True):
            if line.startswith(self.COMMENT_PREFIX) or not line.strip():
                lines.append(line)
            else:
                break
        return "".join(lines)

    def _write(self, by_scope: dict[str, dict[str, str]]) -> None:
        """Serialise deterministically, so re-exporting produces no diff."""
        ordered = {
            scope: dict(sorted(by_scope[scope].items())) for scope in sorted(by_scope)
        }
        body = yaml.safe_dump(
            {self.ALIASES_KEY: ordered},
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
        self._seed_path.parent.mkdir(parents=True, exist_ok=True)
        self._seed_path.write_text(self._header() + body, encoding=self.ENCODING)
