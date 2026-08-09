"""Tests for the openfootball repository sync and corpus source.

The split matters: ``OpenFootballRepository`` is the only piece that touches
the network, and it is driven by an explicit CLI step. The corpus source reads
the checked-out files, so nothing in a training run depends on GitHub.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from config.config_loader import EuropeanConfig
from src.corpus.openfootball.repository import OpenFootballRepository
from src.corpus.openfootball.source import OpenFootballCorpusSource
from src.corpus.supplementary import CorpusSchema, SupplementaryCorpusSource

_CL_2024 = """= UEFA Champions League 2024/25

▪ League, Matchday 1
  Tue Sep 17 2024
    18:45  Sport Lisboa e Benfica (POR) v FC Internazionale Milano (ITA)  2-1 (1-0)
           Arsenal FC (ENG)        v PSV (NED)                0-0
"""

_EL_2024 = """= UEFA Europa League 2024/25

▪ League, Matchday 1
  Wed Sep 25 2024
    18:45  AS Roma (ITA)           v AFC Ajax (NED)           1-1 (0-1)
"""

_CLQ_2024 = """= UEFA Champions League Qualifiers 2024/25

▪ 1. Round
  Tue Jul 09 2024
    18:45  FC Astana (KAZ)         v FC Santa Coloma (AND)    3-0 (2-0)
"""


def _config(root: Path, **overrides) -> EuropeanConfig:
    values = dict(
        enabled=True,
        repo_url="https://github.com/openfootball/champions-league.git",
        checkout_path=str(root),
        competitions={"CL": "cl", "EL": "el", "UECL": "conf"},
        qualifier_competitions={"CLQ": "clq", "ELQ": "elq", "UECLQ": "confq"},
        seasons=["2425"],
        cache_path=str(root / "results.csv"),
        qualifiers_cache_path=str(root / "results_q.csv"),
    )
    values.update(overrides)
    return EuropeanConfig(**values)


def _source_for(config: EuropeanConfig, **kwargs) -> OpenFootballCorpusSource:
    """A source over a real (possibly absent) checkout — no network involved."""
    return OpenFootballCorpusSource(config, OpenFootballRepository(config), **kwargs)


def _checkout(root: Path) -> None:
    season = root / "2024-25"
    season.mkdir(parents=True, exist_ok=True)
    (season / "cl.txt").write_text(_CL_2024, encoding="utf-8")
    (season / "el.txt").write_text(_EL_2024, encoding="utf-8")
    (season / "clq.txt").write_text(_CLQ_2024, encoding="utf-8")


class TestSeasonNaming:
    """football-data season codes and openfootball directories differ."""

    def test_maps_season_code_to_directory(self, tmp_path: Path):
        source = _source_for(_config(tmp_path))
        assert source.season_directory("2425") == "2024-25"

    def test_maps_a_season_spanning_a_decade(self, tmp_path: Path):
        source = _source_for(_config(tmp_path))
        assert source.season_directory("1920") == "2019-20"

    def test_derives_the_calendar_years(self, tmp_path: Path):
        source = _source_for(_config(tmp_path))
        assert source.season_years("2425") == (2024, 2025)


class TestLoad:
    def _source(self, tmp_path: Path, **overrides) -> OpenFootballCorpusSource:
        _checkout(tmp_path)
        return _source_for(_config(tmp_path, **overrides))

    def test_implements_the_supplementary_contract(self, tmp_path: Path):
        assert isinstance(self._source(tmp_path), SupplementaryCorpusSource)

    def test_reads_every_configured_competition(self, tmp_path: Path):
        frame = self._source(tmp_path).load()
        assert set(frame["Div"]) == {"CL", "EL"}

    def test_returns_the_canonical_shape(self, tmp_path: Path):
        frame = self._source(tmp_path).load()
        assert list(frame.columns) == list(CorpusSchema.COLUMNS)

    def test_reads_all_matches(self, tmp_path: Path):
        assert len(self._source(tmp_path).load()) == 3

    def test_missing_competition_file_is_skipped(self, tmp_path: Path):
        """Europa League only exists from 2020-21, Conference from 2021-22."""
        frame = self._source(tmp_path).load()
        assert "UECL" not in set(frame["Div"])

    def test_qualifiers_are_excluded_by_default(self, tmp_path: Path):
        frame = self._source(tmp_path).load()
        assert "CLQ" not in set(frame["Div"])

    def test_qualifiers_included_when_requested(self, tmp_path: Path):
        _checkout(tmp_path)
        config = _config(tmp_path)
        source = OpenFootballCorpusSource(
            config, OpenFootballRepository(config), include_qualifiers=True
        )
        assert "CLQ" in set(source.load()["Div"])

    def test_missing_season_directory_is_skipped(self, tmp_path: Path):
        source = self._source(tmp_path, seasons=["2425", "1718"])
        assert len(source.load()) == 3

    def test_no_checkout_yields_empty_canonical_frame(self, tmp_path: Path):
        source = _source_for(_config(tmp_path / "absent"))
        frame = source.load()
        assert frame.empty
        assert list(frame.columns) == list(CorpusSchema.COLUMNS)

    def test_disabled_config_reads_nothing(self, tmp_path: Path):
        source = self._source(tmp_path, enabled=False)
        assert source.load().empty

    def test_dates_land_in_the_right_season(self, tmp_path: Path):
        frame = self._source(tmp_path).load()
        assert frame["Date"].min() >= pd.Timestamp("2024-07-01")
        assert frame["Date"].max() <= pd.Timestamp("2025-06-30")

    def test_unreadable_file_is_skipped_not_fatal(self, tmp_path: Path):
        _checkout(tmp_path)
        (tmp_path / "2024-25" / "el.txt").write_bytes(b"\xff\xfe\x00bad")
        source = _source_for(_config(tmp_path))
        assert len(source.load()) >= 2


class TestRepositorySync:
    def _repo(self, tmp_path: Path) -> OpenFootballRepository:
        return OpenFootballRepository(_config(tmp_path / "checkout"))

    def test_clones_when_absent(self, tmp_path: Path):
        repo = self._repo(tmp_path)
        with patch("subprocess.run") as run:
            run.return_value = MagicMock(returncode=0)
            assert repo.sync() is True
        assert "clone" in run.call_args[0][0]

    def test_clone_is_shallow(self, tmp_path: Path):
        repo = self._repo(tmp_path)
        with patch("subprocess.run") as run:
            run.return_value = MagicMock(returncode=0)
            repo.sync()
        assert "--depth" in run.call_args[0][0]

    def test_pulls_when_already_checked_out(self, tmp_path: Path):
        checkout = tmp_path / "checkout"
        (checkout / ".git").mkdir(parents=True)
        repo = self._repo(tmp_path)
        with patch("subprocess.run") as run:
            run.return_value = MagicMock(returncode=0)
            repo.sync()
        assert "pull" in run.call_args[0][0]

    def test_failure_returns_false_rather_than_raising(self, tmp_path: Path):
        repo = self._repo(tmp_path)
        with patch("subprocess.run") as run:
            run.return_value = MagicMock(returncode=128, stderr="boom")
            assert repo.sync() is False

    def test_missing_git_returns_false(self, tmp_path: Path):
        repo = self._repo(tmp_path)
        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            assert repo.sync() is False

    def test_timeout_returns_false(self, tmp_path: Path):
        import subprocess

        repo = self._repo(tmp_path)
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("git", 1)
        ):
            assert repo.sync() is False

    def test_exposes_the_checkout_path(self, tmp_path: Path):
        repo = self._repo(tmp_path)
        assert repo.path == tmp_path / "checkout"


class TestSourceErrorBranches:
    def test_malformed_season_code_is_rejected(self, tmp_path: Path):
        import pytest

        source = _source_for(_config(tmp_path))
        with pytest.raises(ValueError):
            source.season_years("24")

    def test_non_numeric_season_code_is_rejected(self, tmp_path: Path):
        import pytest

        source = _source_for(_config(tmp_path))
        with pytest.raises(ValueError):
            source.season_years("abcd")

    def test_malformed_season_in_config_is_skipped_not_fatal(self, tmp_path: Path):
        _checkout(tmp_path)
        source = _source_for(_config(tmp_path, seasons=["bad", "2425"]))
        assert len(source.load()) == 3

    def test_nineties_season_maps_to_the_twentieth_century(self, tmp_path: Path):
        source = _source_for(_config(tmp_path))
        assert source.season_years("9900") == (1999, 2000)
