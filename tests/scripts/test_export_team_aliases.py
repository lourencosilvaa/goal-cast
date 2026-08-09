"""Tests for the Supabase → seed export CLI.

The script exists because CI has no Supabase credentials and must not be given
any. Without the export the 99 approvals are invisible to training, which is
how a green retrain silently produced an uncalibrated model.

Like ``queue_european_team_names.py``, it reads its write back rather than
trusting the call: ``SupabaseTeamAliasRepository`` swallows failures by design
(right for the fixture pipeline, which meets unknown names every run), so a
totally failed export would otherwise look identical to a successful one.
"""

from pathlib import Path

import pytest
import yaml

from src.teams.resolver import TeamAlias

from scripts.export_team_aliases import export_aliases

_APPROVED = [
    TeamAlias("EU-POR", "Sport Lisboa e Benfica", "Benfica"),
    TeamAlias("EU-ENG", "Arsenal FC", "Arsenal"),
]

_HEADER = "# Canonical team-name aliases.\n"


class _Repo:
    def __init__(self, aliases) -> None:
        self._aliases = aliases

    def get_aliases(self):
        return list(self._aliases)


class _FailingRepo:
    def get_aliases(self):
        raise RuntimeError("supabase unreachable")


def _seed(tmp_path: Path, body: str = "aliases: {}\n") -> Path:
    path = tmp_path / "team_aliases.yaml"
    path.write_text(_HEADER + body, encoding="utf-8")
    return path


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["aliases"]


class TestExport:
    def test_writes_approved_aliases_to_the_seed(self, tmp_path: Path):
        path = _seed(tmp_path)
        export_aliases(_Repo(_APPROVED), path)
        assert _read(path)["EU-POR"]["Sport Lisboa e Benfica"] == "Benfica"

    def test_reports_what_it_wrote(self, tmp_path: Path, capsys):
        export_aliases(_Repo(_APPROVED), _seed(tmp_path))
        assert "2" in capsys.readouterr().out

    def test_returns_the_number_written(self, tmp_path: Path):
        assert export_aliases(_Repo(_APPROVED), _seed(tmp_path)) == 2

    def test_an_unreachable_source_raises(self, tmp_path: Path):
        """Silence here is the failure mode the whole change exists to remove."""
        with pytest.raises(RuntimeError):
            export_aliases(_FailingRepo(), _seed(tmp_path))

    def test_an_empty_source_raises_rather_than_reporting_success(
        self, tmp_path: Path
    ):
        """Zero approvals means the read failed or the table is empty.

        Both warrant a non-zero exit: quietly writing nothing is exactly the
        outcome that let the uncalibrated retrain look healthy.
        """
        with pytest.raises(RuntimeError, match="no approved"):
            export_aliases(_Repo([]), _seed(tmp_path))

    def test_the_write_is_verified_by_reading_it_back(self, tmp_path: Path):
        """A seed that does not contain what was exported must fail loudly."""
        path = _seed(tmp_path)
        export_aliases(_Repo(_APPROVED), path)
        from src.teams.resolver import StaticTeamAliasRepository

        assert set(StaticTeamAliasRepository(path).get_aliases()) == set(_APPROVED)

    def test_existing_entries_survive(self, tmp_path: Path):
        path = _seed(tmp_path, 'aliases:\n  P1:\n    "Sporting CP": "Sp Lisbon"\n')
        export_aliases(_Repo(_APPROVED), path)
        assert _read(path)["P1"]["Sporting CP"] == "Sp Lisbon"
