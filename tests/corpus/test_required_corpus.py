"""The corpus may be optional or mandatory, and the caller decides which.

``load_european_corpus`` degrading to an empty frame is right for inference: a
missing cache must not take predictions down, it just means ratings are not
comparable across leagues — exactly as before the corpus existed.

It is wrong for a scheduled retrain. There, degrading quietly means overwriting
a calibrated model with an uncalibrated one and reporting success, which is how
the CI gap went unnoticed. ``european.required`` lets the one loader serve both
without a second code path.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from config.config_loader import EuropeanConfig, TeamAliasConfig, TeamsConfig
from src.corpus import european_corpus
from src.corpus.european_corpus import MissingEuropeanCorpusError, load_european_corpus


class _Config:
    def __init__(self, european) -> None:
        self.european = european
        self.teams = TeamsConfig(aliases=TeamAliasConfig())


def _european(tmp_path: Path, required: bool) -> EuropeanConfig:
    return EuropeanConfig(
        enabled=True,
        required=required,
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
            translated = 0

        self.report = _Report()
        self.report.linkable = linkable or []
        self.report.unlinked_appearances = unlinked


class TestRequired:
    def test_missing_cache_raises(self, tmp_path: Path):
        """Cause 1 of the CI gap: the corpus was never built."""
        with pytest.raises(MissingEuropeanCorpusError):
            load_european_corpus(
                _Config(_european(tmp_path, required=True)), verbose=False
            )

    def test_the_error_names_the_build_script(self, tmp_path: Path):
        with pytest.raises(MissingEuropeanCorpusError, match="build_european_corpus"):
            load_european_corpus(
                _Config(_european(tmp_path, required=True)), verbose=False
            )

    def test_a_corpus_that_links_nothing_raises(self, tmp_path: Path):
        """Cause 2: the corpus is present but no approvals reached it."""
        config = _Config(_european(tmp_path, required=True))
        _write_cache(Path(config.european.cache_path))
        frame = pd.DataFrame({"HomeTeam": ["x"], "AwayTeam": ["y"]})
        with patch.object(european_corpus, "build_translator") as build:
            build.return_value.translate.return_value = _StubResult(frame)
            with pytest.raises(MissingEuropeanCorpusError, match="alias"):
                load_european_corpus(config, verbose=False)

    def test_a_broken_translator_raises(self, tmp_path: Path):
        config = _Config(_european(tmp_path, required=True))
        _write_cache(Path(config.european.cache_path))
        with patch.object(
            european_corpus, "build_translator", side_effect=RuntimeError("no supabase")
        ):
            with pytest.raises(MissingEuropeanCorpusError):
                load_european_corpus(config, verbose=False)

    def test_a_translated_corpus_passes(self, tmp_path: Path):
        config = _Config(_european(tmp_path, required=True))
        _write_cache(Path(config.european.cache_path))
        translated = pd.DataFrame({"HomeTeam": ["Benfica"], "AwayTeam": ["Arsenal"]})
        result = _StubResult(translated)
        result.report.translated = 2
        with patch.object(european_corpus, "build_translator") as build:
            build.return_value.translate.return_value = result
            frame = load_european_corpus(config, verbose=False)
        assert list(frame["HomeTeam"]) == ["Benfica"]


class TestNotRequired:
    """The default must preserve today's behaviour exactly."""

    def test_missing_cache_still_degrades(self, tmp_path: Path):
        assert load_european_corpus(
            _Config(_european(tmp_path, required=False)), verbose=False
        ).empty

    def test_a_broken_translator_still_degrades(self, tmp_path: Path):
        config = _Config(_european(tmp_path, required=False))
        _write_cache(Path(config.european.cache_path))
        with patch.object(
            european_corpus, "build_translator", side_effect=RuntimeError("no supabase")
        ):
            assert load_european_corpus(config, verbose=False).empty

    def test_the_degraded_path_explains_itself(self, tmp_path: Path, capsys):
        """Whoever reads the log must learn why ratings are uncalibrated."""
        config = _Config(_european(tmp_path, required=False))
        _write_cache(Path(config.european.cache_path))
        with patch.object(
            european_corpus, "build_translator", side_effect=RuntimeError("no supabase")
        ):
            load_european_corpus(config, verbose=True)
        assert "no supabase" in capsys.readouterr().out

    def test_a_corpus_that_links_nothing_still_passes_through(self, tmp_path: Path):
        """Untranslated rows are kept — they still build their own ratings."""
        config = _Config(_european(tmp_path, required=False))
        _write_cache(Path(config.european.cache_path))
        frame = pd.DataFrame({"HomeTeam": ["x"], "AwayTeam": ["y"]})
        with patch.object(european_corpus, "build_translator") as build:
            build.return_value.translate.return_value = _StubResult(frame)
            assert not load_european_corpus(config, verbose=False).empty

    def test_required_defaults_to_false(self):
        """An existing config file must keep working unchanged."""
        assert EuropeanConfig().required is False
