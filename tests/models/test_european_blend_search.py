"""Empirical selection of the European blend weight and ELO draw rate.

Both parameters were set by hand: ``dixon_coles_weight`` at 0.6 by analogy
with the domestic blend, ``elo_draw_rate`` at 0.25 as a rough observed rate.
Neither was measured. They are swept together because they interact — a
miscalibrated ELO leg makes ELO look worse than it is and pulls the weight
toward Dixon-Coles.

Two things this must do beyond finding a minimum:

* **Report the baselines.** The useful question is not only "which weight" but
  "does blending beat either model alone, or a constant prior". If it does
  not, that is the finding.
* **Report the uncertainty.** There are roughly 900 evaluable matches. A
  log-loss gap of 0.002 between two cells is noise, and presenting it as a
  result would be the same overstatement the model labels exist to prevent.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from config.config_loader import EuropeanPredictionConfig, EuropeanTuningConfig
from src.models.european_backtest import EuropeanBacktestSet
from src.models.european_blend_search import (
    BlendPoint,
    EuropeanBlendSweeper,
    elo_outcome_from_expected,
)

# Class order used throughout: away=0, draw=1, home=2.
_LABELS = [0, 1, 2]


def _tuning(**overrides: Any) -> EuropeanTuningConfig:
    settings: dict[str, Any] = {
        "blend_weight_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "draw_rate_grid": [0.20, 0.25, 0.30],
        "holdout_seasons": 2,
        "season_start_month": 7,
        "min_holdout_matches": 1,
        "bootstrap_samples": 200,
        "bootstrap_seed": 7,
        "confidence_level": 0.95,
    }
    settings.update(overrides)
    return EuropeanTuningConfig(**settings)


def _dataset(
    elo_expected_home: list[float] | None = None,
    dixon_coles_proba: list[list[float]] | None = None,
    outcomes: list[int] | None = None,
) -> EuropeanBacktestSet:
    """A hand-built evaluation set — no fitting, no corpus, no I/O."""
    expected = np.array(elo_expected_home or [0.6, 0.4, 0.5, 0.7])
    poisson = np.array(
        dixon_coles_proba
        or [
            [0.2, 0.3, 0.5],
            [0.5, 0.3, 0.2],
            [0.3, 0.4, 0.3],
            [0.2, 0.2, 0.6],
        ]
    )
    truth = np.array(outcomes or [2, 0, 1, 2])
    fixtures = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-12-01"] * len(truth)),
            "HomeTeam": [f"H{i}" for i in range(len(truth))],
            "AwayTeam": [f"A{i}" for i in range(len(truth))],
            "fold": ["2024-25"] * len(truth),
        }
    )
    return EuropeanBacktestSet(
        elo_expected_home=expected,
        dixon_coles_proba=poisson,
        outcomes=truth,
        fixtures=fixtures,
        folds=[],
        refused=0,
    )


def _incumbent() -> BlendPoint:
    return BlendPoint(weight=0.5, draw_rate=0.25)


def _sweep(**overrides: Any):
    return EuropeanBlendSweeper(_tuning(**overrides)).sweep(_dataset(), _incumbent())


def _elo_matrix(expected: np.ndarray, draw_rate: float) -> np.ndarray:
    """The ELO leg as [away, draw, home], built the long way for comparison."""
    remaining = 1.0 - draw_rate
    return np.column_stack(
        [
            (1.0 - expected) * remaining,
            np.full(len(expected), draw_rate),
            expected * remaining,
        ]
    )


# ── the shared ELO formula ───────────────────────────────────────────────


class TestEloOutcomeFormula:
    """One formula, used by both the predictor and the sweep.

    ELO yields an expected *score* with draws folded in, so the draw has to be
    carved back out. If the sweep carved it differently from the predictor, it
    would tune a model that is not the one being served — and the two would
    drift apart silently, since nothing downstream compares them.
    """

    def test_it_returns_a_distribution(self):
        probabilities = elo_outcome_from_expected(0.6, 0.25)
        total = probabilities.home_win + probabilities.draw + probabilities.away_win
        assert total == pytest.approx(1.0)

    def test_the_draw_is_the_configured_share(self):
        assert elo_outcome_from_expected(0.6, 0.25).draw == pytest.approx(0.25)

    def test_the_remainder_splits_by_expected_score(self):
        probabilities = elo_outcome_from_expected(0.6, 0.25)
        assert probabilities.home_win == pytest.approx(0.6 * 0.75)
        assert probabilities.away_win == pytest.approx(0.4 * 0.75)

    def test_an_even_fixture_is_symmetric(self):
        probabilities = elo_outcome_from_expected(0.5, 0.3)
        assert probabilities.home_win == pytest.approx(probabilities.away_win)

    def test_the_predictor_uses_this_same_formula(self):
        """Pins the two together so neither can be changed alone."""
        from src.models.european_predictor import EuropeanMatchPredictor

        class StubElo:
            config = type("C", (), {"home_advantage": 0.0})()

            def get_rating(self, team: str) -> float:
                return 1500.0

            def expected_score(self, a: float, b: float) -> float:
                return 0.6

        predictor = EuropeanMatchPredictor(
            dixon_coles=None,
            elo=StubElo(),
            config=EuropeanPredictionConfig(
                enabled=True,
                dixon_coles_weight=0.6,
                min_matches_per_team=1,
                elo_draw_rate=0.25,
                tuning=_tuning(),
            ),
        )
        direct = elo_outcome_from_expected(0.6, 0.25)
        through_predictor = predictor._elo_probabilities("Home", "Away")
        assert through_predictor.home_win == pytest.approx(direct.home_win)
        assert through_predictor.draw == pytest.approx(direct.draw)
        assert through_predictor.away_win == pytest.approx(direct.away_win)


# ── the sweep ────────────────────────────────────────────────────────────


class TestSweepGrid:
    def test_every_combination_is_evaluated(self):
        result = _sweep()
        assert len(result.cells) == 5 * 3

    def test_cells_carry_both_parameters(self):
        result = _sweep()
        assert {(c.weight, c.draw_rate) for c in result.cells} >= {(0.0, 0.20)}

    def test_the_best_cell_has_the_lowest_log_loss(self):
        result = _sweep()
        assert result.best.log_loss == min(c.log_loss for c in result.cells)

    def test_the_best_cell_is_one_of_the_cells(self):
        result = _sweep()
        assert result.best in result.cells

    def test_log_loss_matches_a_hand_computation(self):
        """The arithmetic, pinned against sklearn directly."""
        dataset = _dataset()
        result = _sweep()
        cell = next(c for c in result.cells if c.weight == 0.5 and c.draw_rate == 0.25)
        elo = _elo_matrix(dataset.elo_expected_home, 0.25)
        blended = 0.5 * dataset.dixon_coles_proba + 0.5 * elo
        blended = blended / blended.sum(axis=1, keepdims=True)
        expected = log_loss(dataset.outcomes, blended, labels=_LABELS)
        assert cell.log_loss == pytest.approx(expected)

    def test_weight_zero_is_the_elo_leg_alone(self):
        dataset = _dataset()
        result = _sweep()
        cell = next(c for c in result.cells if c.weight == 0.0 and c.draw_rate == 0.25)
        elo = _elo_matrix(dataset.elo_expected_home, 0.25)
        assert cell.log_loss == pytest.approx(
            log_loss(dataset.outcomes, elo, labels=_LABELS)
        )

    def test_weight_one_is_the_dixon_coles_leg_alone(self):
        dataset = _dataset()
        result = _sweep()
        cell = next(c for c in result.cells if c.weight == 1.0 and c.draw_rate == 0.30)
        assert cell.log_loss == pytest.approx(
            log_loss(dataset.outcomes, dataset.dixon_coles_proba, labels=_LABELS)
        )

    def test_the_draw_rate_does_not_move_the_dixon_coles_only_cells(self):
        """A sanity check on the parameterisation: at w=1 the ELO leg is gone."""
        result = _sweep()
        losses = {c.log_loss for c in result.cells if c.weight == 1.0}
        assert len(losses) == 1

    def test_cells_report_accuracy_and_brier_too(self):
        result = _sweep()
        cell = result.cells[0]
        assert 0.0 <= cell.accuracy <= 1.0
        assert cell.brier > 0.0

    def test_the_grid_order_is_deterministic(self):
        first = [(c.weight, c.draw_rate) for c in _sweep().cells]
        second = [(c.weight, c.draw_rate) for c in _sweep().cells]
        assert first == second


# ── baselines ────────────────────────────────────────────────────────────


class TestBaselines:
    def test_both_legs_and_the_incumbent_are_reported(self):
        result = _sweep()
        assert {b.name for b in result.baselines} >= {
            "elo only",
            "dixon-coles only",
            "base rate",
            "current config",
        }

    def test_the_base_rate_is_a_constant_prediction(self):
        """The bar the blend has to clear to be worth serving at all."""
        result = _sweep()
        base = next(b for b in result.baselines if b.name == "base rate")
        assert base.log_loss > 0.0

    def test_the_current_config_baseline_uses_the_incumbent_values(self):
        result = _sweep()
        current = next(b for b in result.baselines if b.name == "current config")
        cell = next(c for c in result.cells if c.weight == 0.5 and c.draw_rate == 0.25)
        assert current.log_loss == pytest.approx(cell.log_loss)

    def test_an_incumbent_off_the_grid_is_still_evaluated(self):
        """0.6 is not on a 0.25-step grid, and must not silently become 0.5."""
        sweeper = EuropeanBlendSweeper(_tuning())
        result = sweeper.sweep(_dataset(), BlendPoint(weight=0.6, draw_rate=0.25))
        current = next(b for b in result.baselines if b.name == "current config")
        dataset = _dataset()
        elo = _elo_matrix(dataset.elo_expected_home, 0.25)
        blended = 0.6 * dataset.dixon_coles_proba + 0.4 * elo
        blended = blended / blended.sum(axis=1, keepdims=True)
        assert current.log_loss == pytest.approx(
            log_loss(dataset.outcomes, blended, labels=_LABELS)
        )


# ── uncertainty ──────────────────────────────────────────────────────────


class TestBootstrap:
    def test_an_interval_on_the_improvement_is_reported(self):
        low, high = _sweep().improvement_interval
        assert low <= high

    def test_the_interval_is_deterministic_under_a_fixed_seed(self):
        assert _sweep().improvement_interval == _sweep().improvement_interval

    def test_a_different_seed_gives_a_different_interval(self):
        assert (
            _sweep(bootstrap_seed=1).improvement_interval
            != _sweep(bootstrap_seed=2).improvement_interval
        )

    def test_an_interval_spanning_zero_is_not_conclusive(self):
        result = _sweep()
        low, high = result.improvement_interval
        assert result.is_conclusive == (low > 0.0)

    def test_the_incumbent_beating_nothing_is_reported_honestly(self):
        """If the best cell *is* the incumbent, the improvement is zero."""
        sweeper = EuropeanBlendSweeper(
            _tuning(blend_weight_grid=[0.5], draw_rate_grid=[0.25])
        )
        result = sweeper.sweep(_dataset(), BlendPoint(weight=0.5, draw_rate=0.25))
        assert result.improvement == pytest.approx(0.0)
        assert not result.is_conclusive


# ── boundaries ───────────────────────────────────────────────────────────


class TestBoundaries:
    def test_an_empty_dataset_is_refused_rather_than_scored(self):
        empty = EuropeanBacktestSet(
            elo_expected_home=np.array([]),
            dixon_coles_proba=np.empty((0, 3)),
            outcomes=np.array([], dtype=int),
            fixtures=pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "fold"]),
            folds=[],
            refused=0,
        )
        with pytest.raises(ValueError, match="no matches"):
            EuropeanBlendSweeper(_tuning()).sweep(empty, _incumbent())

    def test_a_single_weight_grid_still_sweeps(self):
        result = _sweep(blend_weight_grid=[0.4])
        assert result.best.weight == 0.4

    def test_a_zero_draw_rate_is_allowed(self):
        result = _sweep(draw_rate_grid=[0.0])
        assert all(c.draw_rate == 0.0 for c in result.cells)

    def test_an_empty_weight_grid_is_rejected(self):
        with pytest.raises(ValueError):
            EuropeanTuningConfig(
                blend_weight_grid=[],
                draw_rate_grid=[0.25],
                holdout_seasons=2,
                season_start_month=7,
                min_holdout_matches=1,
                bootstrap_samples=10,
                bootstrap_seed=7,
                confidence_level=0.95,
            )

    def test_a_weight_outside_zero_to_one_is_rejected(self):
        with pytest.raises(ValueError):
            EuropeanTuningConfig(
                blend_weight_grid=[1.5],
                draw_rate_grid=[0.25],
                holdout_seasons=2,
                season_start_month=7,
                min_holdout_matches=1,
                bootstrap_samples=10,
                bootstrap_seed=7,
                confidence_level=0.95,
            )

    def test_a_draw_rate_of_one_is_rejected(self):
        """It would leave no probability for either side to win."""
        with pytest.raises(ValueError):
            EuropeanTuningConfig(
                blend_weight_grid=[0.5],
                draw_rate_grid=[1.0],
                holdout_seasons=2,
                season_start_month=7,
                min_holdout_matches=1,
                bootstrap_samples=10,
                bootstrap_seed=7,
                confidence_level=0.95,
            )

    def test_the_matches_evaluated_are_reported(self):
        assert _sweep().matches == 4
