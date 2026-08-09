"""Boundary and error-path coverage for the alias seed writer.

The seed is the record of every human approval and the only route by which
those approvals reach CI. Losing it silently is the failure this whole change
exists to prevent, so the destructive paths get pinned explicitly.
"""

from pathlib import Path

import pytest
import yaml

from src.teams.alias_seed import AliasSeedWriter
from src.teams.resolver import StaticTeamAliasRepository, TeamAlias

_BENFICA = TeamAlias("EU-POR", "Sport Lisboa e Benfica", "Benfica")


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["aliases"]


class TestMalformedSeed:
    def test_a_scope_that_is_not_a_mapping_raises(self, tmp_path: Path):
        path = tmp_path / "seed.yaml"
        path.write_text("aliases:\n  P1: not-a-mapping\n", encoding="utf-8")
        with pytest.raises(ValueError, match="P1"):
            AliasSeedWriter(path).merge([_BENFICA])

    def test_a_top_level_list_raises(self, tmp_path: Path):
        path = tmp_path / "seed.yaml"
        path.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            AliasSeedWriter(path).merge([_BENFICA])

    def test_unparseable_yaml_raises(self, tmp_path: Path):
        path = tmp_path / "seed.yaml"
        path.write_text("aliases: {\n  unclosed\n", encoding="utf-8")
        with pytest.raises(ValueError, match="valid YAML"):
            AliasSeedWriter(path).merge([_BENFICA])

    def test_a_raising_merge_leaves_the_file_untouched(self, tmp_path: Path):
        """The record must survive a rejected write, not be half-rewritten."""
        path = tmp_path / "seed.yaml"
        original = "aliases:\n  P1: not-a-mapping\n"
        path.write_text(original, encoding="utf-8")
        with pytest.raises(ValueError):
            AliasSeedWriter(path).merge([_BENFICA])
        assert path.read_text(encoding="utf-8") == original


class TestEmptyAndAbsent:
    def test_an_empty_file_is_treated_as_no_entries(self, tmp_path: Path):
        path = tmp_path / "seed.yaml"
        path.write_text("", encoding="utf-8")
        AliasSeedWriter(path).merge([_BENFICA])
        assert _read(path)["EU-POR"]["Sport Lisboa e Benfica"] == "Benfica"

    def test_a_comment_only_file_is_treated_as_no_entries(self, tmp_path: Path):
        path = tmp_path / "seed.yaml"
        path.write_text("# nothing yet\n", encoding="utf-8")
        AliasSeedWriter(path).merge([_BENFICA])
        assert _read(path)["EU-POR"]["Sport Lisboa e Benfica"] == "Benfica"

    def test_an_explicit_null_aliases_key_is_handled(self, tmp_path: Path):
        path = tmp_path / "seed.yaml"
        path.write_text("aliases:\n", encoding="utf-8")
        AliasSeedWriter(path).merge([_BENFICA])
        assert _read(path)["EU-POR"]["Sport Lisboa e Benfica"] == "Benfica"

    def test_a_missing_parent_directory_is_created(self, tmp_path: Path):
        path = tmp_path / "nested" / "deeper" / "seed.yaml"
        AliasSeedWriter(path).merge([_BENFICA])
        assert path.is_file()


class TestValueHandling:
    def test_names_are_stripped(self, tmp_path: Path):
        path = tmp_path / "seed.yaml"
        alias = TeamAlias("  EU-POR  ", "  SL Benfica  ", "  Benfica  ")
        AliasSeedWriter(path).merge([alias])
        assert _read(path)["EU-POR"]["SL Benfica"] == "Benfica"

    def test_a_blank_scope_is_rejected(self, tmp_path: Path):
        path = tmp_path / "seed.yaml"
        AliasSeedWriter(path).merge([TeamAlias("", "SL Benfica", "Benfica")])
        assert _read(path) == {}

    def test_a_blank_canonical_name_is_rejected(self, tmp_path: Path):
        """An alias resolving to nothing would erase a team, not rename it."""
        path = tmp_path / "seed.yaml"
        AliasSeedWriter(path).merge([TeamAlias("EU-POR", "SL Benfica", "  ")])
        assert _read(path) == {}

    def test_non_ascii_names_round_trip(self, tmp_path: Path):
        """Bodø/Glimt and Standard Liège must not be mangled into escapes."""
        path = tmp_path / "seed.yaml"
        alias = TeamAlias("EU-NOR", "FK Bodø/Glimt", "Bodo/Glimt")
        AliasSeedWriter(path).merge([alias])
        assert "Bodø/Glimt" in path.read_text(encoding="utf-8")
        assert StaticTeamAliasRepository(path).get_aliases() == [alias]

    def test_a_duplicate_in_one_batch_keeps_the_last(self, tmp_path: Path):
        path = tmp_path / "seed.yaml"
        first = TeamAlias("EU-POR", "SL Benfica", "Porto")
        second = TeamAlias("EU-POR", "SL Benfica", "Benfica")
        AliasSeedWriter(path).merge([first, second])
        assert _read(path)["EU-POR"]["SL Benfica"] == "Benfica"


class TestShippedSeedIsUsable:
    """The real file, exercised the way training exercises it."""

    def test_the_committed_seed_carries_the_european_approvals(self):
        from config.config_loader import load_config

        config = load_config("config/config.yaml")
        aliases = StaticTeamAliasRepository(config.teams.aliases.seed_path).get_aliases()
        european = [a for a in aliases if a.league_code.startswith("EU-")]
        assert european, (
            "no EU-* aliases in the committed seed — CI cannot reach Supabase, "
            "so run scripts/export_team_aliases.py and commit the result"
        )

    def test_benfica_and_sporting_are_not_confused(self):
        """The mistake the review tool exists to prevent, pinned in the record.

        openfootball's suggester ranks 'Sp Lisbon' first for 'Sport Lisboa e
        Benfica' — and 'Sp Lisbon' is Sporting, not Benfica.
        """
        from config.config_loader import load_config

        config = load_config("config/config.yaml")
        mapping = {
            (a.league_code, a.raw_name): a.canonical_name
            for a in StaticTeamAliasRepository(
                config.teams.aliases.seed_path
            ).get_aliases()
        }
        assert mapping.get(("EU-POR", "Sport Lisboa e Benfica")) == "Benfica"
