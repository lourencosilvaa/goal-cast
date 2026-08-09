"""Tests for the canonical-key → display-name mapping.

The keys stay football-data's ("Sp Lisbon") so nothing has to be remapped or
retrained; only what a person reads changes. A missing mapping must therefore
degrade to today's behaviour — the key itself — never to a blank.
"""

from pathlib import Path

from src.teams.display_names import (
    NullDisplayNameRepository,
    StaticDisplayNameRepository,
)

_SEED = """display_names:
  Sp Lisbon: Sporting Clube de Portugal
  Benfica: Sport Lisboa e Benfica
  Man City: Manchester City FC
"""


def _repository(tmp_path: Path, text: str = _SEED) -> StaticDisplayNameRepository:
    path = tmp_path / "display_names.yaml"
    path.write_text(text, encoding="utf-8")
    return StaticDisplayNameRepository(path)


class TestStaticDisplayNames:
    def test_reads_the_mapping(self, tmp_path: Path):
        assert _repository(tmp_path).all()["Man City"] == "Manchester City FC"

    def test_displays_the_mapped_name(self, tmp_path: Path):
        assert (
            _repository(tmp_path).display_for("Sp Lisbon")
            == "Sporting Clube de Portugal"
        )

    def test_distinguishes_the_two_lisbon_clubs(self, tmp_path: Path):
        """The exact confusion this mapping exists to remove."""
        repository = _repository(tmp_path)
        assert repository.display_for("Sp Lisbon") != repository.display_for("Benfica")

    def test_unmapped_team_shows_its_key(self, tmp_path: Path):
        assert _repository(tmp_path).display_for("Arsenal") == "Arsenal"

    def test_applies_to_a_list_in_order(self, tmp_path: Path):
        result = _repository(tmp_path).apply(["Arsenal", "Man City"])
        assert result == ["Arsenal", "Manchester City FC"]

    def test_missing_file_degrades_to_keys(self, tmp_path: Path):
        repository = StaticDisplayNameRepository(tmp_path / "absent.yaml")
        assert repository.all() == {}
        assert repository.display_for("Arsenal") == "Arsenal"

    def test_malformed_file_degrades_to_keys(self, tmp_path: Path):
        repository = _repository(tmp_path, "display_names: [not, a, mapping]")
        assert repository.all() == {}

    def test_unparseable_yaml_degrades_to_keys(self, tmp_path: Path):
        repository = _repository(tmp_path, "display_names: {unclosed")
        assert repository.all() == {}

    def test_file_without_the_key_degrades_to_keys(self, tmp_path: Path):
        assert _repository(tmp_path, "something_else: {}").all() == {}

    def test_blank_entries_are_ignored(self, tmp_path: Path):
        repository = _repository(
            tmp_path, 'display_names:\n  "": Nameless\n  Arsenal: ""\n'
        )
        assert repository.all() == {}

    def test_file_is_read_once(self, tmp_path: Path):
        repository = _repository(tmp_path)
        first = repository.all()
        (tmp_path / "display_names.yaml").unlink()
        assert repository.all() == first


class TestNullDisplayNames:
    def test_everything_shows_its_key(self):
        assert NullDisplayNameRepository().display_for("Sp Lisbon") == "Sp Lisbon"

    def test_has_no_mappings(self):
        assert NullDisplayNameRepository().all() == {}
