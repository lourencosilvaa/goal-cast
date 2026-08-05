"""Tests for the versioned YAML alias seed.

Entries in this file are human-validated by code review — the second of the
two validation channels, alongside admin approval in the UI. Like every other
repository in the project it degrades to an empty result rather than raising,
so a missing or malformed file can never take the pipeline down.
"""

import pytest

from src.teams.registry import load_team_registry
from src.teams.resolver import StaticTeamAliasRepository, TeamAlias


def write_yaml(tmp_path, text: str) -> str:
    path = tmp_path / "team_aliases.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestStaticTeamAliasRepository:

    def test_reads_aliases_grouped_by_league(self, tmp_path):
        path = write_yaml(
            tmp_path,
            "aliases:\n"
            "  P1:\n"
            '    "Sporting CP": "Sp Lisbon"\n'
            "  E0:\n"
            '    "Manchester City": "Man City"\n',
        )
        aliases = StaticTeamAliasRepository(path).get_aliases()
        assert TeamAlias("P1", "Sporting CP", "Sp Lisbon") in aliases
        assert TeamAlias("E0", "Manchester City", "Man City") in aliases

    def test_missing_file_yields_no_aliases(self, tmp_path):
        assert StaticTeamAliasRepository(tmp_path / "absent.yaml").get_aliases() == []

    def test_malformed_yaml_yields_no_aliases(self, tmp_path):
        path = write_yaml(tmp_path, "aliases: [this is not a mapping")
        assert StaticTeamAliasRepository(path).get_aliases() == []

    def test_empty_file_yields_no_aliases(self, tmp_path):
        assert StaticTeamAliasRepository(write_yaml(tmp_path, "")).get_aliases() == []

    def test_missing_aliases_key_yields_no_aliases(self, tmp_path):
        path = write_yaml(tmp_path, "something_else: 1\n")
        assert StaticTeamAliasRepository(path).get_aliases() == []

    def test_non_mapping_league_entry_is_skipped(self, tmp_path):
        path = write_yaml(
            tmp_path,
            "aliases:\n  P1:\n    - not-a-mapping\n  E0:\n    \"Man Utd\": \"Man United\"\n",
        )
        assert StaticTeamAliasRepository(path).get_aliases() == [
            TeamAlias("E0", "Man Utd", "Man United")
        ]


class TestShippedSeedFile:
    """The seed ships with the image and must always parse."""

    def test_configured_seed_is_readable(self):
        from config.config_loader import load_config

        config = load_config("config/config.yaml")
        # Never raises, whether or not anyone has added entries yet.
        assert isinstance(
            StaticTeamAliasRepository(config.teams.aliases.seed_path).get_aliases(),
            list,
        )

    def test_seed_entries_point_at_real_teams(self):
        """Every committed alias must name a team that exists in the registry."""
        from config.config_loader import load_config

        config = load_config("config/config.yaml")
        registry = load_team_registry(config.teams.registry_path)
        broken = [
            alias
            for alias in StaticTeamAliasRepository(
                config.teams.aliases.seed_path
            ).get_aliases()
            if alias.canonical_name not in registry.get(alias.league_code, [])
        ]
        assert not broken, f"seed aliases naming unknown teams: {broken}"


class TestTeamRegistryLoader:

    def test_loads_teams_by_league(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text('{"E0": ["Arsenal", "Chelsea"]}', encoding="utf-8")
        assert load_team_registry(path) == {"E0": ["Arsenal", "Chelsea"]}

    def test_missing_file_yields_empty_mapping(self, tmp_path):
        assert load_team_registry(tmp_path / "absent.json") == {}

    def test_malformed_json_yields_empty_mapping(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_team_registry(path) == {}

    def test_non_mapping_payload_yields_empty_mapping(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert load_team_registry(path) == {}

    def test_shipped_registry_covers_configured_leagues(self):
        from config.config_loader import load_config

        config = load_config("config/config.yaml")
        registry = load_team_registry(config.teams.registry_path)
        assert set(registry) == set(config.data.leagues)


@pytest.mark.parametrize("raw", ["", "   "])
def test_blank_alias_keys_are_ignored(tmp_path, raw):
    path = tmp_path / "team_aliases.yaml"
    path.write_text(f'aliases:\n  P1:\n    "{raw}": "Sp Lisbon"\n', encoding="utf-8")
    assert StaticTeamAliasRepository(path).get_aliases() == []
