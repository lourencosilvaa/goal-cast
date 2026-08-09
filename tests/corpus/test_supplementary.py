"""Contract tests for the goals-only supplementary corpus.

European results carry no shot, foul or corner columns, so they can only feed
the models that need nothing but goals (ELO, Dixon-Coles) — never the
ensemble's feature matrix, which would drop them anyway. These tests pin the
shape every source must produce so those consumers can trust it blindly.
"""

import inspect

import pandas as pd
import pytest

from src.corpus.supplementary import CorpusSchema, SupplementaryCorpusSource


class _StubSource(SupplementaryCorpusSource):
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def load(self) -> pd.DataFrame:
        return self._frame


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Div": ["CL", "CL"],
            "Season": ["2024-25", "2024-25"],
            "Date": ["2024-09-17", "2024-09-18"],
            "HomeTeam": ["Sport Lisboa e Benfica", "Manchester City FC"],
            "AwayTeam": ["FC Internazionale Milano", "Arsenal FC"],
            "HomeCountry": ["POR", "ENG"],
            "AwayCountry": ["ITA", "ENG"],
            "FTHG": [2, 0],
            "FTAG": [1, 0],
            "HTHG": [1, 0],
            "HTAG": [0, 0],
        }
    )


class TestSupplementaryCorpusSource:
    def test_is_abstract(self):
        assert inspect.isabstract(SupplementaryCorpusSource)

    def test_requires_load(self):
        with pytest.raises(TypeError):
            SupplementaryCorpusSource()  # type: ignore[abstract]


class TestCorpusSchema:
    def test_normalise_produces_canonical_columns(self):
        assert list(CorpusSchema.normalise(_rows()).columns) == list(
            CorpusSchema.COLUMNS
        )

    def test_derives_the_result_from_the_ninety_minute_score(self):
        result = CorpusSchema.normalise(_rows())
        assert list(result["FTR"]) == ["H", "D"]

    def test_derives_an_away_win(self):
        frame = _rows()
        frame.loc[0, "FTHG"], frame.loc[0, "FTAG"] = 0, 3
        assert CorpusSchema.normalise(frame).iloc[0]["FTR"] == "A"

    def test_coerces_dates(self):
        result = CorpusSchema.normalise(_rows())
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_keeps_half_time_goals(self):
        result = CorpusSchema.normalise(_rows())
        assert (result.iloc[0]["HTHG"], result.iloc[0]["HTAG"]) == (1, 0)

    def test_keeps_country_codes(self):
        """Phase 2 needs these to make a name suggestion safe to offer."""
        result = CorpusSchema.normalise(_rows())
        assert result.iloc[0]["HomeCountry"] == "POR"

    def test_missing_half_time_is_tolerated(self):
        frame = _rows().drop(columns=["HTHG", "HTAG"])
        result = CorpusSchema.normalise(frame)
        assert result["HTHG"].isna().all()

    def test_drops_rows_without_goals(self):
        frame = _rows()
        frame.loc[1, "FTHG"] = None
        assert len(CorpusSchema.normalise(frame)) == 1

    def test_drops_rows_without_a_team(self):
        frame = _rows()
        frame.loc[0, "AwayTeam"] = ""
        assert len(CorpusSchema.normalise(frame)) == 1

    def test_drops_rows_with_an_unparseable_date(self):
        frame = _rows()
        frame.loc[0, "Date"] = "not-a-date"
        assert len(CorpusSchema.normalise(frame)) == 1

    def test_strips_team_whitespace(self):
        frame = _rows()
        frame.loc[0, "HomeTeam"] = "  Benfica  "
        assert CorpusSchema.normalise(frame).iloc[0]["HomeTeam"] == "Benfica"

    def test_missing_required_column_yields_empty(self):
        result = CorpusSchema.normalise(_rows().drop(columns=["FTHG"]))
        assert result.empty
        assert list(result.columns) == list(CorpusSchema.COLUMNS)

    def test_empty_input_yields_canonical_columns(self):
        result = CorpusSchema.normalise(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == list(CorpusSchema.COLUMNS)

    def test_none_input_yields_canonical_columns(self):
        assert CorpusSchema.normalise(None).empty  # type: ignore[arg-type]

    def test_empty_helper(self):
        assert list(CorpusSchema.empty().columns) == list(CorpusSchema.COLUMNS)

    def test_from_rows_accepts_parser_output(self):
        rows = [
            {
                "Div": "CL",
                "Season": "2024-25",
                "Date": pd.Timestamp("2024-09-17"),
                "HomeTeam": "Benfica",
                "AwayTeam": "Inter",
                "HomeCountry": "POR",
                "AwayCountry": "ITA",
                "FTHG": 2,
                "FTAG": 1,
                "HTHG": None,
                "HTAG": None,
            }
        ]
        result = CorpusSchema.from_rows(rows)
        assert len(result) == 1
        assert result.iloc[0]["FTR"] == "H"

    def test_from_rows_with_nothing_yields_canonical_columns(self):
        assert list(CorpusSchema.from_rows([]).columns) == list(CorpusSchema.COLUMNS)


class TestChainedCorpusSource:
    """Fallback sources accumulate; a dead one contributes nothing."""

    def _chained(self, *sources):
        from src.corpus.supplementary import ChainedCorpusSource

        return ChainedCorpusSource(list(sources))

    def _broken(self):
        class _Broken(SupplementaryCorpusSource):
            def load(self) -> pd.DataFrame:
                raise RuntimeError("source unavailable")

        return _Broken()

    def test_concatenates_distinct_matches(self):
        other = _rows()
        other.loc[0, "Date"] = "2024-10-01"
        result = self._chained(_StubSource(_rows()), _StubSource(other)).load()
        assert len(result) == 3

    def test_deduplicates_the_same_match_from_two_sources(self):
        """Feeding one result twice would double its weight in the ratings."""
        result = self._chained(_StubSource(_rows()), _StubSource(_rows())).load()
        assert len(result) == 2

    def test_deduplication_can_be_turned_off(self):
        result = self._chained(_StubSource(_rows()), _StubSource(_rows())).load(
            deduplicate=False
        )
        assert len(result) == 4

    def test_skips_a_failing_source(self):
        result = self._chained(self._broken(), _StubSource(_rows())).load()
        assert len(result) == 2

    def test_all_sources_failing_yields_canonical_columns(self):
        result = self._chained(self._broken(), self._broken()).load()
        assert result.empty
        assert list(result.columns) == list(CorpusSchema.COLUMNS)

    def test_no_sources_yields_canonical_columns(self):
        result = self._chained().load()
        assert result.empty
        assert list(result.columns) == list(CorpusSchema.COLUMNS)

    def test_output_is_normalised(self):
        result = self._chained(_StubSource(_rows().drop(columns=["HTHG"]))).load()
        assert list(result.columns) == list(CorpusSchema.COLUMNS)


class TestStaticFileCorpusSource:
    """The cache reader that keeps the network out of a training run."""

    def _source(self, path):
        from src.corpus.supplementary import StaticFileCorpusSource

        return StaticFileCorpusSource(path)

    def test_reads_csv(self, tmp_path):
        path = tmp_path / "results.csv"
        CorpusSchema.normalise(_rows()).to_csv(path, index=False)
        assert len(self._source(path).load()) == 2

    def test_reads_parquet(self, tmp_path):
        path = tmp_path / "results.parquet"
        CorpusSchema.normalise(_rows()).to_parquet(path, index=False)
        assert len(self._source(path).load()) == 2

    def test_missing_file_yields_canonical_columns(self, tmp_path):
        result = self._source(tmp_path / "absent.csv").load()
        assert result.empty
        assert list(result.columns) == list(CorpusSchema.COLUMNS)

    def test_malformed_file_yields_empty(self, tmp_path):
        path = tmp_path / "results.parquet"
        path.write_text("not parquet at all", encoding="utf-8")
        assert self._source(path).load().empty

    def test_unknown_suffix_yields_empty_rather_than_guessing(self, tmp_path):
        path = tmp_path / "results.txt"
        path.write_text("Div,Date\nCL,2024-09-17\n", encoding="utf-8")
        assert self._source(path).load().empty

    def test_directory_at_the_path_yields_empty(self, tmp_path):
        directory = tmp_path / "results.csv"
        directory.mkdir()
        assert self._source(directory).load().empty

    def test_accepts_a_string_path(self, tmp_path):
        path = tmp_path / "results.csv"
        CorpusSchema.normalise(_rows()).to_csv(path, index=False)
        assert len(self._source(str(path)).load()) == 2

    def test_exposes_its_path(self, tmp_path):
        from pathlib import Path

        assert self._source(tmp_path / "results.csv").path == Path(
            tmp_path / "results.csv"
        )
