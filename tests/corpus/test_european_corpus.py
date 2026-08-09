"""Tests for the shared European corpus loader.

Training and all three inference paths call this. If they diverged, the model
would be trained on calibrated ELO features and served uncalibrated ones —
worse than never calibrating, because the features stop meaning what the model
learned them to mean.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from config.config_loader import EuropeanConfig, TeamAliasConfig, TeamsConfig
from src.corpus import european_corpus
from src.corpus.european_corpus import load_european_corpus


class _Config:
    def __init__(self, european, teams=None) -> None:
        if european is not None:
            self.european = european
        self.teams = teams or TeamsConfig(aliases=TeamAliasConfig())


def _european(tmp_path: Path, enabled: bool = True) -> EuropeanConfig:
    return EuropeanConfig(
        enabled=enabled,
        cache_path=str(tmp_path / "results.csv"),
        country_leagues={"POR": ["P1"]},
        alias_scope="EU",
    )


def _write_cache(path: Path) -> None:
    pd.DataFrame(
        {
            "Div": ["CL"],
            "Season": ["2024-25"],
            "Date": ["2024-09-17"],
            "HomeTeam": ["Sport Lisboa e Benfica"],
            "AwayTeam": ["Arsenal"],
            "HomeCountry": ["POR"],
            "AwayCountry": ["ENG"],
            "FTHG": [2],
            "FTAG": [1],
            "FTR": ["H"],
            "HTHG": [1],
            "HTAG": [0],
        }
    ).to_csv(path, index=False)


class _StubResult:
    def __init__(self, frame, linkable=None, unlinked=0) -> None:
        self.frame = frame

        class _Report:
            translated = 2
            pass

        self.report = _Report()
        self.report.linkable = linkable or []
        self.report.unlinked_appearances = unlinked


class TestLoading:
    def test_reads_and_translates_the_cache(self, tmp_path: Path):
        config = _Config(_european(tmp_path))
        _write_cache(Path(config.european.cache_path))
        translated = pd.DataFrame({"HomeTeam": ["Benfica"], "AwayTeam": ["Arsenal"]})
        with patch.object(european_corpus, "build_translator") as build:
            build.return_value.translate.return_value = _StubResult(translated)
            result = load_european_corpus(config, verbose=False)
        assert list(result["HomeTeam"]) == ["Benfica"]

    def test_missing_cache_yields_empty(self, tmp_path: Path):
        assert load_european_corpus(_Config(_european(tmp_path)), verbose=False).empty

    def test_disabled_track_yields_empty(self, tmp_path: Path):
        config = _Config(_european(tmp_path, enabled=False))
        _write_cache(Path(config.european.cache_path))
        assert load_european_corpus(config, verbose=False).empty

    def test_config_without_a_european_section_yields_empty(self, tmp_path: Path):
        """An older config file must not break training or inference."""
        assert load_european_corpus(_Config(None), verbose=False).empty

    def test_a_broken_translator_degrades_rather_than_raising(self, tmp_path: Path):
        config = _Config(_european(tmp_path))
        _write_cache(Path(config.european.cache_path))
        with patch.object(
            european_corpus, "build_translator", side_effect=RuntimeError("no supabase")
        ):
            assert load_european_corpus(config, verbose=False).empty


class TestReporting:
    def test_warns_when_no_corpus_is_present(self, tmp_path: Path, capsys):
        load_european_corpus(_Config(_european(tmp_path)), verbose=True)
        assert "NOT be comparable" in capsys.readouterr().out

    def test_reports_unapproved_names(self, tmp_path: Path, capsys):
        config = _Config(_european(tmp_path))
        _write_cache(Path(config.european.cache_path))
        frame = pd.DataFrame({"HomeTeam": ["x"], "AwayTeam": ["y"]})
        with patch.object(european_corpus, "build_translator") as build:
            build.return_value.translate.return_value = _StubResult(
                frame, linkable=["Sport Lisboa e Benfica"], unlinked=12
            )
            load_european_corpus(config, verbose=True)
        out = capsys.readouterr().out
        assert "still unapproved" in out and "12" in out

    def test_quiet_mode_prints_nothing(self, tmp_path: Path, capsys):
        load_european_corpus(_Config(_european(tmp_path)), verbose=False)
        assert capsys.readouterr().out == ""
