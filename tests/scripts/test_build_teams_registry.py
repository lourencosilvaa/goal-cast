"""Phase 1 tests for the teams-registry builder script.

The builder turns the raw per-season CSV cache into the static registry file
shipped inside the backend image. All inputs are written explicitly per test.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.build_teams_registry import (  # noqa: E402
    TeamsRegistryBuilder,
    TeamsRegistrySpec,
    _parse_args,
    main,
)

_HEADER = "Div,Date,HomeTeam,AwayTeam,FTR\n"


def _write_season(cache_dir: Path, season: str, league: str, rows: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{season}_{league}.csv").write_text(_HEADER + rows, encoding="utf-8")


def _spec(cache_dir: Path, output_path: Path) -> TeamsRegistrySpec:
    return TeamsRegistrySpec(
        cache_dir=cache_dir,
        output_path=output_path,
        league_codes=["E0", "SP1"],
    )


class TestTeamsRegistryBuilder:

    def test_builds_teams_per_league(self, tmp_path):
        cache = tmp_path / "cache"
        _write_season(
            cache, "2526", "E0", "E0,01/09/2025,Arsenal,Chelsea,H\n"
        )
        _write_season(
            cache, "2526", "SP1", "SP1,01/09/2025,Barcelona,Sevilla,A\n"
        )
        builder = TeamsRegistryBuilder(_spec(cache, tmp_path / "out.json"))
        assert builder.build() == {
            "E0": ["Arsenal", "Chelsea"],
            "SP1": ["Barcelona", "Sevilla"],
        }

    def test_uses_only_the_latest_season(self, tmp_path):
        cache = tmp_path / "cache"
        _write_season(cache, "2425", "E0", "E0,01/09/2024,Relegated FC,Chelsea,H\n")
        _write_season(cache, "2526", "E0", "E0,01/09/2025,Arsenal,Chelsea,H\n")
        builder = TeamsRegistryBuilder(_spec(cache, tmp_path / "out.json"))
        assert builder.build()["E0"] == ["Arsenal", "Chelsea"]

    def test_missing_cache_dir_builds_empty_registry(self, tmp_path):
        builder = TeamsRegistryBuilder(_spec(tmp_path / "absent", tmp_path / "o.json"))
        assert builder.build() == {}

    def test_league_without_cache_file_is_omitted(self, tmp_path):
        cache = tmp_path / "cache"
        _write_season(cache, "2526", "E0", "E0,01/09/2025,Arsenal,Chelsea,H\n")
        builder = TeamsRegistryBuilder(_spec(cache, tmp_path / "out.json"))
        assert list(builder.build()) == ["E0"]

    def test_csv_without_team_columns_is_omitted(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir(parents=True)
        (cache / "2526_E0.csv").write_text("Div,Date\nE0,01/09/2025\n", encoding="utf-8")
        builder = TeamsRegistryBuilder(_spec(cache, tmp_path / "out.json"))
        assert builder.build() == {}

    def test_blank_and_whitespace_team_names_are_dropped(self, tmp_path):
        cache = tmp_path / "cache"
        _write_season(
            cache,
            "2526",
            "E0",
            "E0,01/09/2025,Arsenal,Chelsea,H\nE0,08/09/2025,  ,,\n",
        )
        builder = TeamsRegistryBuilder(_spec(cache, tmp_path / "out.json"))
        assert builder.build() == {"E0": ["Arsenal", "Chelsea"]}

    def test_header_only_csv_is_omitted(self, tmp_path):
        cache = tmp_path / "cache"
        _write_season(cache, "2526", "E0", "")
        builder = TeamsRegistryBuilder(_spec(cache, tmp_path / "out.json"))
        assert builder.build() == {}

    def test_cache_dir_that_is_a_file_builds_empty(self, tmp_path):
        not_a_dir = tmp_path / "cache"
        not_a_dir.write_text("", encoding="utf-8")
        builder = TeamsRegistryBuilder(_spec(not_a_dir, tmp_path / "out.json"))
        assert builder.build() == {}

    def test_undecodable_csv_is_skipped(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir(parents=True)
        (cache / "2526_E0.csv").write_bytes(b"Div,HomeTeam,AwayTeam\n\xff\xfe,A,B\n")
        _write_season(cache, "2526", "SP1", "SP1,01/09/2025,Barcelona,Sevilla,A\n")
        builder = TeamsRegistryBuilder(_spec(cache, tmp_path / "out.json"))
        assert builder.build() == {"SP1": ["Barcelona", "Sevilla"]}

    def test_write_creates_missing_parent_directories(self, tmp_path):
        cache = tmp_path / "cache"
        _write_season(cache, "2526", "E0", "E0,01/09/2025,Arsenal,Chelsea,H\n")
        output = tmp_path / "nested" / "deeper" / "teams.json"
        builder = TeamsRegistryBuilder(_spec(cache, output))
        builder.write(builder.build())
        assert output.exists()

    def test_non_ascii_team_names_survive_a_round_trip(self, tmp_path):
        cache = tmp_path / "cache"
        _write_season(cache, "2526", "E0", "E0,01/09/2025,Málaga,Nîmes,H\n")
        output = tmp_path / "teams.json"
        builder = TeamsRegistryBuilder(_spec(cache, output))
        builder.write(builder.build())
        assert json.loads(output.read_text(encoding="utf-8"))["E0"] == [
            "Málaga",
            "Nîmes",
        ]

    def test_write_persists_sorted_json(self, tmp_path):
        cache = tmp_path / "cache"
        _write_season(
            cache,
            "2526",
            "E0",
            "E0,01/09/2025,Chelsea,Arsenal,H\nE0,08/09/2025,Arsenal,Fulham,D\n",
        )
        output = tmp_path / "teams_registry.json"
        builder = TeamsRegistryBuilder(_spec(cache, output))
        builder.write(builder.build())
        assert json.loads(output.read_text(encoding="utf-8")) == {
            "E0": ["Arsenal", "Chelsea", "Fulham"]
        }


#: Values ``main()`` actually reads. Everything else in the config file is
#: irrelevant here, so the repo schema is reused and these keys overridden —
#: the test never depends on the repo's own league map or registry path.
_TEST_LEAGUES = {"E0": "Premier League", "SP1": "La Liga"}


def _write_config(tmp_path: Path, registry_path: Path) -> Path:
    """Write a config file with every value this test depends on set explicitly."""
    import yaml

    from config.config_loader import Config

    source = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["data"]["leagues"] = dict(_TEST_LEAGUES)
    data["teams"] = {"registry_path": str(registry_path)}

    Config(**data)  # fail loudly if the schema drifts
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["build_teams_registry.py"])
        args = _parse_args()
        assert args.config == "config/config.yaml"
        assert args.cache_dir == "datasets/cache"
        assert args.output == ""

    def test_overrides(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "build_teams_registry.py",
                "--config", "c.yaml",
                "--cache-dir", "cache",
                "--output", "o.json",
            ],
        )
        args = _parse_args()
        assert (args.config, args.cache_dir, args.output) == ("c.yaml", "cache", "o.json")


class TestMain:
    def test_writes_registry_for_configured_leagues(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        _write_season(cache, "2526", "E0", "E0,01/09/2025,Arsenal,Chelsea,H\n")
        _write_season(cache, "2526", "SP1", "SP1,01/09/2025,Barcelona,Sevilla,A\n")
        registry = tmp_path / "teams.json"
        config = _write_config(tmp_path, registry)
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_teams_registry.py", "--config", str(config),
             "--cache-dir", str(cache)],
        )
        main()
        assert json.loads(registry.read_text(encoding="utf-8")) == {
            "E0": ["Arsenal", "Chelsea"],
            "SP1": ["Barcelona", "Sevilla"],
        }

    def test_output_flag_overrides_config_path(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        _write_season(cache, "2526", "E0", "E0,01/09/2025,Arsenal,Chelsea,H\n")
        config = _write_config(tmp_path, tmp_path / "from_config.json")
        override = tmp_path / "from_flag.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_teams_registry.py", "--config", str(config),
             "--cache-dir", str(cache), "--output", str(override)],
        )
        main()
        assert override.exists()
        assert not (tmp_path / "from_config.json").exists()

    def test_empty_cache_writes_nothing_and_warns(self, tmp_path, monkeypatch, capsys):
        registry = tmp_path / "teams.json"
        config = _write_config(tmp_path, registry)
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_teams_registry.py", "--config", str(config),
             "--cache-dir", str(tmp_path / "absent")],
        )
        main()
        assert "WARNING" in capsys.readouterr().out
        assert not registry.exists()
