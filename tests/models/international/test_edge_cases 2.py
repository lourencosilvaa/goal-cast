from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.config_loader import (
    EloConfig,
    FatigueConfig,
    FeaturesConfig,
    InternationalConfig,
    XgProxyConfig,
)
from src.models.international.data_loader import InternationalDataLoader
from src.models.international.feature_engineer import InternationalFeatureEngineer


def _features_config() -> FeaturesConfig:
    return FeaturesConfig(
        rolling_window=5,
        elo=EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0),
        xg_proxy=XgProxyConfig(sot_conversion=0.3, shot_conversion=0.03),
        fatigue=FatigueConfig(
            max_rest_days=30, default_rest_days=14, fatigue_threshold=3
        ),
    )


def _loader(tmp_path: Path, df: pd.DataFrame, **overrides) -> InternationalDataLoader:
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    cfg = InternationalConfig(
        dataset_path=str(path),
        min_date=overrides.get("min_date", "1800-01-01"),
        tournaments=overrides.get("tournaments", []),
    )
    return InternationalDataLoader(cfg)


class TestDataLoaderEdge:
    def test_malformed_scores_are_dropped(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "home_team": ["A", "B"],
                "away_team": ["C", "D"],
                "home_score": ["x", 2],
                "away_score": [1, 0],
                "tournament": ["Friendly", "Friendly"],
                "neutral": [False, False],
            }
        )
        loader = _loader(tmp_path, df)
        out = loader.load_all()
        assert len(out) == 1
        assert set(out["HomeTeam"]) == {"B"}

    def test_unknown_tournament_filter_yields_empty(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "home_team": ["A"],
                "away_team": ["B"],
                "home_score": [1],
                "away_score": [0],
                "tournament": ["Friendly"],
                "neutral": [False],
            }
        )
        loader = _loader(tmp_path, df, tournaments=["Nonexistent Cup"])
        assert loader.load_all().empty

    def test_missing_neutral_column_defaults_false(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "home_team": ["A"],
                "away_team": ["B"],
                "home_score": [1],
                "away_score": [0],
                "tournament": ["Friendly"],
            }
        )
        loader = _loader(tmp_path, df)
        out = loader.load_all()
        assert bool(out["Neutral"].iloc[0]) is False

    def test_string_true_false_neutral_parsing(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "home_team": ["A", "B"],
                "away_team": ["C", "D"],
                "home_score": [1, 2],
                "away_score": [0, 2],
                "tournament": ["Friendly", "Friendly"],
                "neutral": ["TRUE", "FALSE"],
            }
        )
        loader = _loader(tmp_path, df)
        out = loader.load_all().sort_values("Date").reset_index(drop=True)
        assert bool(out.loc[0, "Neutral"]) is True
        assert bool(out.loc[1, "Neutral"]) is False

    def test_away_win_result_derived(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "home_team": ["A"],
                "away_team": ["B"],
                "home_score": [0],
                "away_score": [3],
                "tournament": ["Friendly"],
                "neutral": [False],
            }
        )
        loader = _loader(tmp_path, df)
        out = loader.load_all()
        assert out["FTR"].iloc[0] == "A"

    def test_read_error_returns_empty(self, tmp_path: Path):
        # Point the dataset path at a directory so pandas raises on read.
        cfg = InternationalConfig(dataset_path=str(tmp_path))
        loader = InternationalDataLoader(cfg)
        assert loader.load_all().empty

    def test_all_invalid_rows_returns_empty(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "home_team": ["A"],
                "away_team": ["B"],
                "home_score": ["x"],
                "away_score": ["y"],
                "tournament": ["Friendly"],
                "neutral": [False],
            }
        )
        loader = _loader(tmp_path, df)
        assert loader.load_all().empty


class TestFeatureEngineerEdge:
    def test_all_neutral_pool_builds_features(self):
        teams = ["A", "B", "C", "D"]
        rng = np.random.default_rng(3)
        base = pd.Timestamp("2016-01-01")
        rows = []
        for i in range(40):
            h, a = rng.choice(teams, size=2, replace=False)
            hg, ag = int(rng.integers(0, 3)), int(rng.integers(0, 3))
            ftr = "H" if hg > ag else ("A" if ag > hg else "D")
            rows.append(
                {
                    "Date": base + pd.Timedelta(days=7 * i),
                    "HomeTeam": h,
                    "AwayTeam": a,
                    "FTHG": hg,
                    "FTAG": ag,
                    "FTR": ftr,
                    "Neutral": True,
                    "Tournament": "FIFA World Cup",
                }
            )
        df = pd.DataFrame(rows)
        out = InternationalFeatureEngineer(_features_config()).build_all_features(df)
        assert not out.empty
        assert (out["is_neutral"] == 1).all()

    def test_too_few_matches_drops_early_rows(self):
        # A team with < min_periods (3) prior matches yields NaN rolling
        # stats, so build_match_features drops those rows.
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2020-01-01", "2020-01-08", "2020-01-15"]
                ),
                "HomeTeam": ["A", "A", "A"],
                "AwayTeam": ["B", "B", "B"],
                "FTHG": [1, 2, 0],
                "FTAG": [0, 2, 1],
                "FTR": ["H", "D", "A"],
                "Neutral": [False, False, False],
                "Tournament": ["Friendly", "Friendly", "Friendly"],
            }
        )
        out = InternationalFeatureEngineer(_features_config()).build_all_features(df)
        assert len(out) < len(df)
