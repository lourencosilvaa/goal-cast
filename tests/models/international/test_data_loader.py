from pathlib import Path

import pandas as pd
import pytest

from config.config_loader import InternationalConfig
from src.models.international.base import AbstractMatchDataLoader
from src.models.international.data_loader import InternationalDataLoader


def _make_config(tmp_path: Path, **overrides) -> InternationalConfig:
    csv_path = overrides.pop("dataset_path", str(tmp_path / "results.csv"))
    data = {
        "enabled": True,
        "dataset_path": csv_path,
        "min_date": "1800-01-01",
        "models_dir": "output/models/international",
        "neutral_home_advantage_factor": 0.0,
        "tournaments": [],
        "flashscore": {"leagues": {"WC": "world/world-cup"}},
    }
    data.update(overrides)
    return InternationalConfig(**data)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


class TestInternationalDataLoader:
    def test_implements_abstract_loader(self, tmp_path: Path, raw_international_rows):
        _write_csv(tmp_path / "results.csv", raw_international_rows)
        loader = InternationalDataLoader(_make_config(tmp_path))
        assert isinstance(loader, AbstractMatchDataLoader)

    def test_normalizes_to_canonical_schema(
        self, tmp_path: Path, raw_international_rows
    ):
        _write_csv(tmp_path / "results.csv", raw_international_rows)
        loader = InternationalDataLoader(_make_config(tmp_path))
        df = loader.load_all()
        for col in (
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
            "Neutral",
            "Tournament",
        ):
            assert col in df.columns

    def test_derives_full_time_result(self, tmp_path: Path, raw_international_rows):
        _write_csv(tmp_path / "results.csv", raw_international_rows)
        loader = InternationalDataLoader(_make_config(tmp_path))
        df = loader.load_all().sort_values("Date").reset_index(drop=True)
        # Russia 5-0 Saudi -> H ; Portugal 3-3 Spain -> D ; Brazil 1-1 -> D
        result_by_home = dict(zip(df["HomeTeam"], df["FTR"]))
        assert result_by_home["Russia"] == "H"
        assert result_by_home["Portugal"] == "D"
        assert result_by_home["Spain"] == "H"

    def test_neutral_parsed_as_bool(self, tmp_path: Path, raw_international_rows):
        _write_csv(tmp_path / "results.csv", raw_international_rows)
        loader = InternationalDataLoader(_make_config(tmp_path))
        df = loader.load_all()
        assert df["Neutral"].dtype == bool
        neutral_map = dict(zip(df["HomeTeam"], df["Neutral"]))
        assert neutral_map["Portugal"] is True or bool(neutral_map["Portugal"]) is True
        assert bool(neutral_map["Scotland"]) is False

    def test_min_date_filter(self, tmp_path: Path, raw_international_rows):
        _write_csv(tmp_path / "results.csv", raw_international_rows)
        loader = InternationalDataLoader(
            _make_config(tmp_path, min_date="2000-01-01")
        )
        df = loader.load_all()
        assert (pd.to_datetime(df["Date"]) >= pd.Timestamp("2000-01-01")).all()
        assert "Scotland" not in set(df["HomeTeam"])  # 1872 row filtered out

    def test_tournament_filter(self, tmp_path: Path, raw_international_rows):
        _write_csv(tmp_path / "results.csv", raw_international_rows)
        loader = InternationalDataLoader(
            _make_config(tmp_path, tournaments=["FIFA World Cup"])
        )
        df = loader.load_all()
        assert set(df["Tournament"].unique()) == {"FIFA World Cup"}

    def test_empty_tournaments_means_all(
        self, tmp_path: Path, raw_international_rows
    ):
        _write_csv(tmp_path / "results.csv", raw_international_rows)
        loader = InternationalDataLoader(_make_config(tmp_path, tournaments=[]))
        df = loader.load_all()
        assert {"Friendly", "FIFA World Cup", "UEFA Euro"} <= set(df["Tournament"])

    def test_missing_file_returns_empty(self, tmp_path: Path):
        loader = InternationalDataLoader(
            _make_config(tmp_path, dataset_path=str(tmp_path / "nope.csv"))
        )
        df = loader.load_all()
        assert df.empty
