"""Tests for the European corpus build CLI."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from config.config_loader import EuropeanConfig

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "build_european_corpus.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_european_corpus", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_european_corpus"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _config() -> EuropeanConfig:
    return EuropeanConfig(
        enabled=True,
        repo_url="https://example.test/openfootball.git",
        checkout_path="datasets/openfootball/champions-league",
        competitions={"CL": "cl", "EL": "el", "UECL": "conf"},
        qualifier_competitions={"CLQ": "clq"},
        seasons=["2526", "2425", "2324"],
        cache_path="datasets/european/results.csv",
        qualifiers_cache_path="datasets/european/results_with_qualifiers.csv",
    )


class TestNarrow:
    def test_no_flag_keeps_every_season(self, script):
        assert script._narrow(_config(), None).seasons == ["2526", "2425", "2324"]

    def test_selects_a_subset(self, script):
        assert script._narrow(_config(), "2425").seasons == ["2425"]

    def test_tolerates_whitespace(self, script):
        assert script._narrow(_config(), " 2425 , 2324 ").seasons == ["2425", "2324"]

    def test_unknown_season_selects_nothing_rather_than_guessing(self, script):
        assert script._narrow(_config(), "9999").seasons == []

    def test_leaves_the_original_config_untouched(self, script):
        config = _config()
        script._narrow(config, "2425")
        assert len(config.seasons) == 3


class TestWrite:
    def test_creates_missing_parent_directories(self, script, tmp_path: Path):
        target = tmp_path / "nested" / "deep" / "results.csv"
        script._write(pd.DataFrame({"Div": ["CL"]}), str(target))
        assert target.is_file()

    def test_writes_without_an_index_column(self, script, tmp_path: Path):
        target = tmp_path / "results.csv"
        script._write(pd.DataFrame({"Div": ["CL"], "FTHG": [2]}), str(target))
        assert list(pd.read_csv(target).columns) == ["Div", "FTHG"]


class TestReport:
    def test_empty_frame_reports_without_raising(self, script, capsys):
        script._report(pd.DataFrame(), "Main draws")
        assert "0 matches" in capsys.readouterr().out

    def test_summarises_each_competition(self, script, capsys):
        frame = pd.DataFrame(
            {
                "Div": ["CL", "CL", "EL"],
                "Season": ["2425", "2425", "2425"],
                "Date": pd.to_datetime(["2024-09-17", "2024-10-01", "2024-09-25"]),
            }
        )
        script._report(frame, "Main draws")
        out = capsys.readouterr().out
        assert "CL" in out and "EL" in out
