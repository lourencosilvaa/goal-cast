"""Tests for MatchStatsCalculator's Dixon-Coles integration.

When a fitted Dixon-Coles model is supplied and knows both teams, the
goal-based markets are derived from its calibrated scoreline distribution.
Otherwise the calculator falls back to the naive independent-Poisson path.
"""

import numpy as np
import pandas as pd
import pytest

from config.config_loader import DataConfig, PoissonConfig
from src.analysis.match_stats import (
    MatchStats,
    MatchStatsCalculator,
    _poisson_prob,
)
from src.models.poisson.dixon_coles import DixonColesModel


def _data_config() -> DataConfig:
    # Empty seasons => _load_league_data returns an empty frame without any
    # disk access, isolating the market computation under test.
    return DataConfig(
        base_url="http://example.test",
        seasons=[],
        leagues={"E0": "Premier League"},
        columns_to_keep=[],
    )


def _fitted_model() -> DixonColesModel:
    rng = np.random.default_rng(0)
    teams = ["Strong", "Weak"]
    attack = {"Strong": 0.6, "Weak": -0.6}
    defense = {"Strong": -0.5, "Weak": 0.5}
    rows = []
    date = pd.Timestamp("2022-01-01")
    for _ in range(40):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lam = np.exp(attack[home] + defense[away] + 0.3)
                mu = np.exp(attack[away] + defense[home])
                rows.append(
                    {
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "FTHG": int(rng.poisson(lam)),
                        "FTAG": int(rng.poisson(mu)),
                        "Date": date,
                    }
                )
                date += pd.Timedelta(days=1)
    cfg = PoissonConfig(
        enabled=True, max_goals=8, half_life_days=3650, blend_weight=0.4
    )
    return DixonColesModel(cfg).fit(pd.DataFrame(rows))


class TestHelpers:
    def test_poisson_prob_zero_lambda(self) -> None:
        assert _poisson_prob(0.0, 0) == 1.0
        assert _poisson_prob(0.0, 2) == 0.0

    def test_matchstats_to_dict_shape(self) -> None:
        ms = MatchStats(
            home_team="A",
            away_team="B",
            league="L",
            over25_prob=0.55,
            btts_yes_prob=0.6,
            btts_no_prob=0.4,
            top_scorelines=[(1, 0, 0.2), (2, 1, 0.1)],
        )
        d = ms.to_dict()
        assert d["over_under"]["over_2.5"] == 0.55
        assert d["btts"]["yes"] == 0.6
        assert d["top_scorelines"][0] == {"score": "1-0", "prob": 0.2}


class TestDixonColesMarkets:
    def test_markets_come_from_model_when_known(self) -> None:
        model = _fitted_model()
        calc = MatchStatsCalculator(_data_config(), poisson_model=model)
        stats = calc.compute_match_stats("E0", "Strong", "Weak")

        dc = model.predict("Strong", "Weak")
        assert stats.home_xg == pytest.approx(dc.lambda_home)
        assert stats.away_xg == pytest.approx(dc.lambda_away)
        assert stats.over25_prob == pytest.approx(dc.over_25)
        assert stats.over15_prob == pytest.approx(dc.over_15)
        assert stats.btts_yes_prob == pytest.approx(dc.btts_yes)
        assert stats.top_scorelines == dc.top_scorelines


def _synthetic_league() -> pd.DataFrame:
    """A small league with the columns the naive calculator reads."""
    rng = np.random.default_rng(3)
    teams = ["Alpha", "Beta", "Gamma"]
    rows = []
    date = pd.Timestamp("2023-01-01")
    for _ in range(12):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                rows.append(
                    {
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "FTHG": int(rng.integers(0, 4)),
                        "FTAG": int(rng.integers(0, 3)),
                        "HS": float(rng.integers(5, 20)),
                        "AS": float(rng.integers(4, 18)),
                        "HST": float(rng.integers(2, 9)),
                        "AST": float(rng.integers(1, 8)),
                        "HC": float(rng.integers(1, 11)),
                        "AC": float(rng.integers(1, 10)),
                        "Date": date,
                    }
                )
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


class TestNaiveFallback:
    def _calc_with_data(self) -> MatchStatsCalculator:
        calc = MatchStatsCalculator(_data_config(), poisson_model=None)
        # Inject data directly so the naive path runs without disk access.
        calc._data_cache["E0"] = _synthetic_league()
        return calc

    def test_naive_path_computes_valid_markets(self) -> None:
        calc = self._calc_with_data()
        stats = calc.compute_match_stats("E0", "Alpha", "Beta")
        assert stats.home_xg > 0.0
        assert stats.away_xg > 0.0
        assert 0.0 <= stats.over25_prob <= 1.0
        assert stats.btts_yes_prob + stats.btts_no_prob == pytest.approx(1.0, abs=1e-6)
        assert len(stats.top_scorelines) > 0

    def test_team_stats_populated_from_history(self) -> None:
        calc = self._calc_with_data()
        team = calc.compute_team_stats("E0", "Alpha")
        assert team.matches_played > 0
        assert team.avg_shots > 0.0
        assert 0.0 <= team.btts_pct <= 1.0

    def test_falls_back_when_no_model_empty_data(self) -> None:
        calc = MatchStatsCalculator(_data_config(), poisson_model=None)
        stats = calc.compute_match_stats("E0", "Strong", "Weak")
        assert stats.home_xg > 0.0
        assert 0.0 <= stats.over25_prob <= 1.0

    def test_falls_back_when_team_unknown(self) -> None:
        model = _fitted_model()
        calc = MatchStatsCalculator(_data_config(), poisson_model=model)
        # "Ghost" is unseen => naive path, not the Dixon-Coles branch.
        stats = calc.compute_match_stats("E0", "Strong", "Ghost")
        dc = model.predict("Strong", "Ghost")
        # Naive xG differs from the model's lambda for the unseen matchup.
        assert stats.home_xg != pytest.approx(dc.lambda_home)
        assert stats.home_xg > 0.0

    def test_empty_team_stats_for_unknown_team(self) -> None:
        calc = self._calc_with_data()
        team = calc.compute_team_stats("E0", "DoesNotExist")
        assert team.matches_played == 0
