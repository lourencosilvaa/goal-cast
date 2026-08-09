"""Tests for projecting approved aliases into the committed seed file.

The approvals that calibrate cross-league ratings live in Supabase, which CI
cannot reach — it is given no credentials, by design, because the network has
no business in the training path. So the scheduled retrain saw *zero* of the 99
approvals and silently produced an uncalibrated model while reporting success.

This writer is what closes that: Supabase stays the review surface, the seed
becomes the record. Everything training needs is then on disk and in git.

The merge is deliberately one-way and additive. Entries already in the seed
that the export does not mention are left alone — a hand-written FlashScore
alias must survive an export of European approvals.
"""

from pathlib import Path

import pytest
import yaml

from src.teams.alias_seed import AliasSeedWriter, SeedMergeReport
from src.teams.resolver import StaticTeamAliasRepository, TeamAlias

_BENFICA = TeamAlias(
    league_code="EU-POR", raw_name="Sport Lisboa e Benfica", canonical_name="Benfica"
)
_SPORTING = TeamAlias(
    league_code="EU-POR", raw_name="Sporting Clube de Portugal", canonical_name="Sp Lisbon"
)
_ARSENAL = TeamAlias(
    league_code="EU-ENG", raw_name="Arsenal FC", canonical_name="Arsenal"
)

_HEADER = "# Canonical team-name aliases — validated by code review.\n#\n# Second line.\n"


def _seed(tmp_path: Path, body: str = "aliases: {}\n", header: str = _HEADER) -> Path:
    path = tmp_path / "team_aliases.yaml"
    path.write_text(header + body, encoding="utf-8")
    return path


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["aliases"]


class TestMerge:
    def test_writes_an_approved_alias(self, tmp_path: Path):
        path = _seed(tmp_path)
        AliasSeedWriter(path).merge([_BENFICA])
        assert _read(path)["EU-POR"]["Sport Lisboa e Benfica"] == "Benfica"

    def test_groups_by_scope(self, tmp_path: Path):
        path = _seed(tmp_path)
        AliasSeedWriter(path).merge([_BENFICA, _ARSENAL])
        aliases = _read(path)
        assert set(aliases) == {"EU-POR", "EU-ENG"}

    def test_keeps_several_entries_in_one_scope(self, tmp_path: Path):
        path = _seed(tmp_path)
        AliasSeedWriter(path).merge([_BENFICA, _SPORTING])
        assert len(_read(path)["EU-POR"]) == 2

    def test_existing_entries_are_not_clobbered(self, tmp_path: Path):
        """A hand-written alias must survive an export it was not part of."""
        path = _seed(tmp_path, 'aliases:\n  P1:\n    "Sporting CP": "Sp Lisbon"\n')
        AliasSeedWriter(path).merge([_BENFICA])
        aliases = _read(path)
        assert aliases["P1"]["Sporting CP"] == "Sp Lisbon"
        assert aliases["EU-POR"]["Sport Lisboa e Benfica"] == "Benfica"

    def test_an_existing_mapping_is_updated(self, tmp_path: Path):
        path = _seed(
            tmp_path, 'aliases:\n  EU-POR:\n    "Sport Lisboa e Benfica": "Sp Lisbon"\n'
        )
        AliasSeedWriter(path).merge([_BENFICA])
        assert _read(path)["EU-POR"]["Sport Lisboa e Benfica"] == "Benfica"

    def test_is_idempotent(self, tmp_path: Path):
        """Re-exporting must not churn the file — the diff is the review."""
        path = _seed(tmp_path)
        writer = AliasSeedWriter(path)
        writer.merge([_BENFICA, _ARSENAL])
        first = path.read_text(encoding="utf-8")
        writer.merge([_BENFICA, _ARSENAL])
        assert path.read_text(encoding="utf-8") == first

    def test_output_is_ordered_deterministically(self, tmp_path: Path):
        """Same aliases in any order must produce the same bytes."""
        forward = tmp_path / "forward.yaml"
        forward.write_text(_HEADER + "aliases: {}\n", encoding="utf-8")
        backward = tmp_path / "backward.yaml"
        backward.write_text(_HEADER + "aliases: {}\n", encoding="utf-8")

        AliasSeedWriter(forward).merge([_BENFICA, _ARSENAL, _SPORTING])
        AliasSeedWriter(backward).merge([_SPORTING, _ARSENAL, _BENFICA])
        assert forward.read_text(encoding="utf-8") == backward.read_text(
            encoding="utf-8"
        )

    def test_empty_export_leaves_the_seed_intact(self, tmp_path: Path):
        """A Supabase read that returns nothing must never truncate the record."""
        path = _seed(tmp_path, 'aliases:\n  P1:\n    "Sporting CP": "Sp Lisbon"\n')
        AliasSeedWriter(path).merge([])
        assert _read(path)["P1"]["Sporting CP"] == "Sp Lisbon"


class TestHeaderPreservation:
    def test_the_explanatory_header_survives(self, tmp_path: Path):
        """The header documents the human-validation rule; dumping YAML loses it."""
        path = _seed(tmp_path)
        AliasSeedWriter(path).merge([_BENFICA])
        assert path.read_text(encoding="utf-8").startswith(_HEADER)

    def test_a_seed_without_a_header_still_works(self, tmp_path: Path):
        path = _seed(tmp_path, header="")
        AliasSeedWriter(path).merge([_BENFICA])
        assert _read(path)["EU-POR"]["Sport Lisboa e Benfica"] == "Benfica"


class TestReport:
    def test_counts_additions(self, tmp_path: Path):
        report = AliasSeedWriter(_seed(tmp_path)).merge([_BENFICA, _ARSENAL])
        assert isinstance(report, SeedMergeReport)
        assert report.added == 2

    def test_counts_unchanged_entries(self, tmp_path: Path):
        path = _seed(tmp_path)
        writer = AliasSeedWriter(path)
        writer.merge([_BENFICA])
        assert writer.merge([_BENFICA]).unchanged == 1

    def test_counts_updates(self, tmp_path: Path):
        path = _seed(
            tmp_path, 'aliases:\n  EU-POR:\n    "Sport Lisboa e Benfica": "Sp Lisbon"\n'
        )
        assert AliasSeedWriter(path).merge([_BENFICA]).updated == 1

    def test_total_counts_everything_in_the_seed(self, tmp_path: Path):
        path = _seed(tmp_path, 'aliases:\n  P1:\n    "Sporting CP": "Sp Lisbon"\n')
        assert AliasSeedWriter(path).merge([_BENFICA]).total == 2


class TestRoundTrip:
    def test_what_is_written_is_what_the_resolver_reads(self, tmp_path: Path):
        """The whole point: the seed must feed StaticTeamAliasRepository."""
        path = _seed(tmp_path)
        AliasSeedWriter(path).merge([_BENFICA, _ARSENAL, _SPORTING])
        loaded = StaticTeamAliasRepository(path).get_aliases()
        assert set(loaded) == {_BENFICA, _ARSENAL, _SPORTING}


class TestRobustness:
    def test_a_missing_seed_file_is_created(self, tmp_path: Path):
        path = tmp_path / "absent.yaml"
        AliasSeedWriter(path).merge([_BENFICA])
        assert _read(path)["EU-POR"]["Sport Lisboa e Benfica"] == "Benfica"

    def test_blank_names_are_rejected(self, tmp_path: Path):
        path = _seed(tmp_path)
        blank = TeamAlias(league_code="EU-POR", raw_name="  ", canonical_name="Benfica")
        AliasSeedWriter(path).merge([blank])
        assert _read(path) == {}

    def test_a_malformed_seed_raises_rather_than_silently_resetting(
        self, tmp_path: Path
    ):
        """Overwriting an unparseable seed would destroy the record it holds."""
        path = tmp_path / "seed.yaml"
        path.write_text("aliases: [this is not a mapping]\n", encoding="utf-8")
        with pytest.raises(ValueError):
            AliasSeedWriter(path).merge([_BENFICA])
