"""Tests for _predict_fixture league-average fallback in run_inference.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.run_inference import _predict_fixture


def _make_featured_data() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
            "home_Form": 2.0, "away_Form": 1.5,
            "home_avg_GF": 2.1, "away_avg_GF": 1.3,
            "home_avg_GA": 0.9, "away_avg_GA": 1.2,
            "elo_home": 1600.0, "elo_away": 1500.0, "elo_diff": 100.0,
            "elo_expected_home": 0.64, "elo_expected_away": 0.36,
            "home_xG_rolling": 2.0, "away_xG_rolling": 1.0,
            "home_rest_days": 7.0, "away_rest_days": 5.0,
            "home_draw_pct": 0.25, "away_draw_pct": 0.30,
            "h2h_home_wins": 0.5, "h2h_draws": 0.3,
            "xG_diff": 1.0, "diff_Form": 0.5, "diff_avg_GF": 0.8,
            "rest_advantage": 2.0, "avg_draw_pct": 0.275,
            "form_gap": 0.5, "attack_similarity": 0.7,
            "defense_similarity": 0.6, "combined_defensive": 1.2,
            "is_midweek": 0,
        },
        {
            "HomeTeam": "Liverpool", "AwayTeam": "Man City",
            "home_Form": 2.5, "away_Form": 2.8,
            "home_avg_GF": 2.5, "away_avg_GF": 2.7,
            "home_avg_GA": 0.8, "away_avg_GA": 0.7,
            "elo_home": 1700.0, "elo_away": 1750.0, "elo_diff": -50.0,
            "elo_expected_home": 0.46, "elo_expected_away": 0.54,
            "home_xG_rolling": 2.5, "away_xG_rolling": 2.7,
            "home_rest_days": 6.0, "away_rest_days": 6.0,
            "home_draw_pct": 0.20, "away_draw_pct": 0.22,
            "h2h_home_wins": 0.4, "h2h_draws": 0.35,
            "xG_diff": -0.2, "diff_Form": -0.3, "diff_avg_GF": -0.2,
            "rest_advantage": 0.0, "avg_draw_pct": 0.21,
            "form_gap": 0.3, "attack_similarity": 0.85,
            "defense_similarity": 0.9, "combined_defensive": 1.4,
            "is_midweek": 1,
        },
    ])


def _make_fixture(home: str = "Arsenal", away: str = "Chelsea"):
    f = MagicMock()
    f.home_team = home
    f.away_team = away
    f.b365_home = 0.0
    f.b365_draw = 0.0
    f.b365_away = 0.0
    return f


def _make_predictor(feature_names: list[str]):
    pred = MagicMock()
    pred.feature_names = feature_names
    result = MagicMock()
    result.predicted_outcome = "Home Win"
    result.confidence = 0.65
    pred.predict.return_value = [result]
    return pred


class TestPredictFixtureLeagueAverageFallback:

    def test_known_both_teams_uses_model(self):
        df = _make_featured_data()
        predictor = _make_predictor(["home_Form", "away_Form", "elo_diff"])
        fixture = _make_fixture("Arsenal", "Chelsea")

        result = _predict_fixture(fixture, df, predictor)

        predictor.predict.assert_called_once()
        assert result.predicted_outcome == "Home Win"

    def test_unknown_home_team_still_calls_model(self):
        """Newly promoted home team: uses league averages, still calls the ML model."""
        df = _make_featured_data()
        predictor = _make_predictor(["home_Form", "away_Form", "elo_diff"])
        fixture = _make_fixture("Promoted FC", "Arsenal")

        result = _predict_fixture(fixture, df, predictor)

        predictor.predict.assert_called_once()

    def test_unknown_away_team_still_calls_model(self):
        """Newly promoted away team: uses league averages, still calls the ML model."""
        df = _make_featured_data()
        predictor = _make_predictor(["home_Form", "away_Form", "elo_diff"])
        fixture = _make_fixture("Arsenal", "Relegated FC")

        result = _predict_fixture(fixture, df, predictor)

        predictor.predict.assert_called_once()

    def test_both_teams_unknown_still_calls_model(self):
        """Both teams have no history: league averages used for both."""
        df = _make_featured_data()
        predictor = _make_predictor(["home_Form", "away_Form", "elo_diff"])
        fixture = _make_fixture("New FC", "Unknown United")

        result = _predict_fixture(fixture, df, predictor)

        predictor.predict.assert_called_once()

    def test_empty_featured_data_falls_back_to_odds(self):
        """No historical data at all: must fall back to odds-based prediction."""
        df = pd.DataFrame()
        predictor = _make_predictor(["home_Form"])
        fixture = _make_fixture("Arsenal", "Chelsea")

        result = _predict_fixture(fixture, df, predictor)

        predictor.predict.assert_not_called()

    def test_league_average_produces_valid_feature_row(self):
        """Feature row built from league averages must have correct numeric values."""
        df = _make_featured_data()
        captured_df: list = []

        predictor = _make_predictor(["home_Form", "away_Form"])
        predictor.predict.side_effect = lambda d: (captured_df.append(d), [MagicMock()])[1]

        _predict_fixture(_make_fixture("Unknown FC", "Arsenal"), df, predictor)

        assert len(captured_df) == 1
        row = captured_df[0].iloc[0]
        expected_avg_form = df["home_Form"].mean()
        assert abs(row["home_Form"] - expected_avg_form) < 1e-6
