"""Tests for the Dixon-Coles Poisson score model.

The model must recover sensible attack/defense strengths from synthetic
data generated with known parameters, and derive coherent 1X2, O/U 2.5 and
BTTS markets from the fitted scoreline distribution.
"""

import numpy as np
import pandas as pd
import pytest

from config.config_loader import PoissonConfig
from src.models.outcome_model import OutcomeProbabilities
from src.models.poisson.dixon_coles import DixonColesModel


def _synthetic_matches(rounds: int = 40) -> pd.DataFrame:
    """Generate matches with a dominant, a mid and a weak team."""
    rng = np.random.default_rng(0)
    teams = ["Strong", "Mid", "Weak"]
    attack = {"Strong": 0.6, "Mid": 0.0, "Weak": -0.6}
    defense = {"Strong": -0.5, "Mid": 0.0, "Weak": 0.5}  # higher = concedes more
    gamma = 0.3
    rows = []
    date = pd.Timestamp("2022-01-01")
    for _ in range(rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lam = np.exp(attack[home] + defense[away] + gamma)
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
    return pd.DataFrame(rows)


def _make_config(max_goals: int = 8, blend_weight: float = 0.4) -> PoissonConfig:
    return PoissonConfig(
        enabled=True,
        max_goals=max_goals,
        half_life_days=3650,
        blend_weight=blend_weight,
    )


class TestDixonColesFit:
    def test_recovers_relative_strengths(self) -> None:
        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        assert model.attack["Strong"] > model.attack["Mid"]
        assert model.attack["Mid"] > model.attack["Weak"]
        # Lower defense value = concedes fewer goals.
        assert model.defense["Strong"] < model.defense["Weak"]

    def test_home_advantage_positive(self) -> None:
        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        assert model.home_advantage > 0.0


class TestDixonColesPredict:
    def test_outcome_probs_sum_to_one(self) -> None:
        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        pred = model.predict("Strong", "Weak")
        total = pred.home_win + pred.draw + pred.away_win
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_stronger_home_team_favoured(self) -> None:
        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        strong_home = model.predict("Strong", "Weak")
        weak_home = model.predict("Weak", "Strong")
        assert strong_home.home_win > weak_home.home_win

    def test_markets_in_unit_interval(self) -> None:
        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        pred = model.predict("Strong", "Mid")
        for value in (
            pred.over_15,
            pred.over_25,
            pred.over_35,
            pred.under_25,
            pred.btts_yes,
            pred.btts_no,
        ):
            assert 0.0 <= value <= 1.0

    def test_over_under_monotonic_and_complementary(self) -> None:
        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        pred = model.predict("Strong", "Mid")
        # More goals required => lower probability.
        assert pred.over_15 >= pred.over_25 >= pred.over_35
        # under_2.5 and over_2.5 partition the sample space.
        assert pred.under_25 + pred.over_25 == pytest.approx(1.0, abs=1e-6)
        assert pred.btts_yes + pred.btts_no == pytest.approx(1.0, abs=1e-6)

    def test_expected_goals_positive(self) -> None:
        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        pred = model.predict("Strong", "Weak")
        assert pred.lambda_home > 0.0
        assert pred.lambda_away > 0.0
        # The dominant home side is expected to outscore the weak away side.
        assert pred.lambda_home > pred.lambda_away

    def test_knows_seen_and_unseen_teams(self) -> None:
        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        assert model.knows("Strong")
        assert not model.knows("Nonexistent United")

    def test_top_scorelines_sorted_desc(self) -> None:
        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        pred = model.predict("Strong", "Weak")
        probs = [p for _, _, p in pred.top_scorelines]
        assert probs == sorted(probs, reverse=True)
        assert len(pred.top_scorelines) >= 1

    def test_as_outcome_returns_outcome_probabilities(self) -> None:
        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        outcome = model.predict_outcome("Strong", "Weak")
        assert isinstance(outcome, OutcomeProbabilities)
        assert outcome.home_win + outcome.draw + outcome.away_win == pytest.approx(
            1.0, abs=1e-6
        )

    def test_unseen_team_falls_back_without_error(self) -> None:
        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        pred = model.predict("Nonexistent United", "Weak")
        total = pred.home_win + pred.draw + pred.away_win
        assert total == pytest.approx(1.0, abs=1e-6)


class TestDixonColesNuances:
    def test_non_positive_half_life_uses_uniform_weights(self) -> None:
        cfg = PoissonConfig(
            enabled=True, max_goals=6, half_life_days=0.0, blend_weight=0.4
        )
        model = DixonColesModel(cfg)
        weights = model._time_weights(_synthetic_matches(rounds=2)["Date"])
        assert np.allclose(weights, 1.0)

    def test_max_goals_one_still_valid_distribution(self) -> None:
        cfg = PoissonConfig(
            enabled=True, max_goals=1, half_life_days=3650, blend_weight=0.4
        )
        model = DixonColesModel(cfg).fit(_synthetic_matches())
        pred = model.predict("Strong", "Weak")
        assert pred.home_win + pred.draw + pred.away_win == pytest.approx(
            1.0, abs=1e-6
        )
        # With max 1 goal each, BTTS is only the 1-1 scoreline.
        assert 0.0 <= pred.btts_yes <= 1.0

    def test_top_scorelines_capped(self) -> None:
        model = DixonColesModel(_make_config(max_goals=8)).fit(_synthetic_matches())
        pred = model.predict("Strong", "Weak")
        assert len(pred.top_scorelines) == DixonColesModel._TOP_SCORELINES

    def test_both_teams_unseen_uses_home_advantage(self) -> None:
        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        pred = model.predict("Ghost A", "Ghost B")
        # Equal (zero) strengths => home edge from home advantage only.
        assert pred.home_win > pred.away_win


class TestDixonColesPersistence:
    def test_save_load_round_trip(self, tmp_path) -> None:
        import joblib

        model = DixonColesModel(_make_config()).fit(_synthetic_matches())
        path = tmp_path / "poisson_model.joblib"
        joblib.dump(model, path)
        reloaded = joblib.load(path)

        before = model.predict("Strong", "Weak")
        after = reloaded.predict("Strong", "Weak")
        assert after.home_win == pytest.approx(before.home_win)
