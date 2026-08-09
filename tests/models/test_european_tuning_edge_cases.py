"""Edge cases found while implementing the European blend sweep.

Separated from the two behaviour suites because these are the boundaries the
implementation revealed rather than the behaviour it was written to satisfy:
degenerate corpora, teams that appear only in the holdout, folds that survive
nothing, and the paths a real run will hit on a config that predates the
tuning section entirely.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from config.config_loader import (
    EloConfig,
    EuropeanPredictionConfig,
    EuropeanTuningConfig,
)
from src.models.cross_competition import CrossCompetitionCorpus
from src.models.elo import FootballELO
from src.models.european_backtest import (
    BacktestModels,
    BacktestFold,
    EuropeanBacktester,
    EuropeanBacktestSet,
    SeasonSplitter,
)
from src.models.european_blend_search import (
    BlendPoint,
    EuropeanBlendSweeper,
    elo_outcome_from_expected,
)
from src.models.outcome_model import OutcomeProbabilities


class StubDixonColes:
    def __init__(self) -> None:
        self.fitted: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> "StubDixonColes":
        self.fitted = df.copy()
        return self

    def knows(self, team: str) -> bool:
        if self.fitted is None:
            return False
        return team in set(self.fitted["HomeTeam"]) | set(self.fitted["AwayTeam"])

    def predict_outcome(self, home_team: str, away_team: str) -> OutcomeProbabilities:
        return OutcomeProbabilities(0.45, 0.28, 0.27).normalized()


def _elo_config() -> EloConfig:
    return EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0)


def _tuning(**overrides: Any) -> EuropeanTuningConfig:
    settings: dict[str, Any] = {
        "blend_weight_grid": [0.0, 0.5, 1.0],
        "draw_rate_grid": [0.25],
        "holdout_seasons": 2,
        "season_start_month": 7,
        "min_holdout_matches": 1,
        "bootstrap_samples": 25,
        "bootstrap_seed": 3,
        "confidence_level": 0.9,
    }
    settings.update(overrides)
    return EuropeanTuningConfig(**settings)


def _prediction(**overrides: Any) -> EuropeanPredictionConfig:
    settings: dict[str, Any] = {
        "enabled": True,
        "dixon_coles_weight": 0.6,
        "min_matches_per_team": 1,
        "elo_draw_rate": 0.25,
        "tuning": _tuning(),
    }
    settings.update(overrides)
    return EuropeanPredictionConfig(**settings)


def _domestic() -> pd.DataFrame:
    """Three seasons, so the earliest fold has something to train on.

    Two would not: a fold whose cut precedes all domestic history refuses
    every fixture, correctly but uninformatively, and would mask whatever the
    test was actually about.
    """
    rows = []
    for year in (2022, 2023, 2024):
        rows.extend(
            [
                ("E0", f"{year}-10-01", "Arsenal", "Everton", 2, 0),
                ("P1", f"{year}-10-02", "Benfica", "Boavista", 2, 1),
            ]
        )
    frame = pd.DataFrame(
        rows, columns=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    )
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["League"] = frame["Div"]
    return frame


def _supplementary(rows: list[tuple] | None = None) -> pd.DataFrame:
    rows = rows or [
        ("CL", "2023-12-01", "Benfica", "Arsenal", 1, 2),
        ("CL", "2024-12-01", "Arsenal", "Benfica", 3, 1),
    ]
    frame = pd.DataFrame(
        rows, columns=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    )
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


def _run(
    domestic: pd.DataFrame | None = None,
    supplementary: pd.DataFrame | None = None,
    prediction: EuropeanPredictionConfig | None = None,
    **run_kwargs: Any,
) -> EuropeanBacktestSet:
    backtester = EuropeanBacktester(
        corpus=CrossCompetitionCorpus(
            domestic=_domestic() if domestic is None else domestic,
            supplementary=_supplementary() if supplementary is None else supplementary,
        ),
        prediction=prediction or _prediction(),
        models=BacktestModels(
            elo=lambda: FootballELO(_elo_config()),
            dixon_coles=StubDixonColes,
        ),
    )
    return backtester.run(**run_kwargs)


# ── degenerate corpora ───────────────────────────────────────────────────


class TestDegenerateCorpora:
    def test_no_domestic_history_yields_nothing(self):
        """European rows alone cannot calibrate anything to compare against."""
        assert _run(domestic=pd.DataFrame()).is_empty

    def test_no_european_history_yields_nothing(self):
        assert _run(supplementary=pd.DataFrame()).is_empty

    def test_a_single_european_match_still_produces_a_fold(self):
        single = _supplementary([("CL", "2024-12-01", "Arsenal", "Benfica", 1, 0)])
        result = _run(supplementary=single)
        assert len(result) == 1

    def test_european_matches_all_in_one_season_give_one_fold(self):
        same_season = _supplementary(
            [
                ("CL", "2024-09-01", "Arsenal", "Benfica", 1, 0),
                ("CL", "2025-02-01", "Benfica", "Arsenal", 0, 0),
            ]
        )
        result = _run(supplementary=same_season)
        assert len(result.folds) == 1
        assert result.folds[0].label == "2024-25"

    def test_a_european_match_before_any_domestic_one_is_still_scored(self):
        early = _supplementary([("CL", "2024-08-01", "Arsenal", "Benfica", 1, 0)])
        result = _run(supplementary=early)
        assert len(result) == 1


# ── teams at the edge of the corpus ──────────────────────────────────────


class TestTeamsAtTheEdge:
    def test_a_team_appearing_only_in_the_holdout_is_refused(self):
        """It has no history before the cut, so nothing could predict it."""
        with_newcomer = _supplementary(
            [
                ("CL", "2023-12-01", "Benfica", "Arsenal", 1, 2),
                ("CL", "2024-12-01", "Arsenal", "Newcomer FC", 3, 1),
            ]
        )
        result = _run(supplementary=with_newcomer)
        assert "Newcomer FC" not in set(result.fixtures["AwayTeam"])
        assert result.refused == 1

    def test_a_team_known_only_from_europe_can_still_be_evaluated(self):
        """Cross-league links are the point; a European-only club has history."""
        european_only = _supplementary(
            [
                ("CL", "2023-12-01", "Arsenal", "Galatasaray", 1, 0),
                ("CL", "2024-12-01", "Galatasaray", "Arsenal", 2, 2),
            ]
        )
        result = _run(supplementary=european_only)
        assert "Galatasaray" in set(result.fixtures["HomeTeam"])

    def test_a_goalless_draw_encodes_as_a_draw(self):
        goalless = _supplementary(
            [
                ("CL", "2023-12-01", "Benfica", "Arsenal", 1, 2),
                ("CL", "2024-12-01", "Arsenal", "Benfica", 0, 0),
            ]
        )
        result = _run(supplementary=goalless)
        last = result.fixtures["Date"] == pd.Timestamp("2024-12-01")
        assert int(result.outcomes[last.to_numpy()][0]) == 1


# ── folds ────────────────────────────────────────────────────────────────


class TestFoldReporting:
    def test_a_skipped_fold_still_appears_in_the_report(self):
        """Silently dropping it would hide how little evidence there was."""
        prediction = _prediction(tuning=_tuning(min_holdout_matches=5))
        result = _run(prediction=prediction)
        assert len(result.folds) == 2
        assert all(fold.evaluated == 0 for fold in result.folds)

    def test_refusals_survive_a_skipped_fold(self):
        prediction = _prediction(
            min_matches_per_team=99, tuning=_tuning(min_holdout_matches=5)
        )
        result = _run(prediction=prediction)
        assert result.refused == 2

    def test_max_folds_beyond_what_exists_is_harmless(self):
        assert len(_run(max_folds=99).folds) == 2

    def test_a_negative_max_folds_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            _run(max_folds=-1)

    def test_fold_labels_cross_the_century_correctly(self):
        fold = BacktestFold(
            season_start_year=2099,
            cut=pd.Timestamp("2099-07-01"),
            end=pd.Timestamp("2100-07-01"),
        )
        assert fold.label == "2099-00"

    def test_a_january_cut_month_keeps_calendar_seasons(self):
        """Some competitions do run on calendar years; the config allows it."""
        folds = SeasonSplitter(
            _tuning(holdout_seasons=1, season_start_month=1)
        ).folds(pd.Series(pd.to_datetime(["2024-03-01"])))
        assert folds[0].cut == pd.Timestamp("2024-01-01")

    def test_dates_that_cannot_be_parsed_are_dropped_not_fatal(self):
        dates = pd.Series(pd.to_datetime(["2024-12-01", None]))
        folds = SeasonSplitter(_tuning(holdout_seasons=1)).folds(dates)
        assert len(folds) == 1


# ── the sweeper's numerics ───────────────────────────────────────────────


def _dataset(n: int = 6) -> EuropeanBacktestSet:
    rng = np.random.default_rng(0)
    expected = rng.uniform(0.2, 0.8, size=n)
    poisson = rng.dirichlet([2.0, 2.0, 2.0], size=n)
    return EuropeanBacktestSet(
        elo_expected_home=expected,
        dixon_coles_proba=poisson,
        outcomes=rng.integers(0, 3, size=n),
        fixtures=pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-12-01"] * n),
                "HomeTeam": [f"H{i}" for i in range(n)],
                "AwayTeam": [f"A{i}" for i in range(n)],
                "fold": ["2024-25"] * n,
            }
        ),
        folds=[],
        refused=0,
    )


class TestSweeperNumerics:
    def test_every_cell_is_a_finite_number(self):
        result = EuropeanBlendSweeper(_tuning()).sweep(
            _dataset(), BlendPoint(0.6, 0.25)
        )
        assert all(np.isfinite(cell.log_loss) for cell in result.cells)

    def test_a_certain_and_wrong_leg_does_not_produce_infinity(self):
        """Blending is what keeps a zero out of the log; the sweep must show it.

        At weight 1.0 the Dixon-Coles leg is used alone, and a leg that put
        zero probability on what happened would give an infinite loss. That is
        a real property of the model, not an artefact, so it must be reported
        rather than clipped away.
        """
        dataset = EuropeanBacktestSet(
            elo_expected_home=np.array([0.5, 0.5]),
            dixon_coles_proba=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            outcomes=np.array([2, 2]),
            fixtures=pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2024-12-01"] * 2),
                    "HomeTeam": ["H0", "H1"],
                    "AwayTeam": ["A0", "A1"],
                    "fold": ["2024-25"] * 2,
                }
            ),
            folds=[],
            refused=0,
        )
        result = EuropeanBlendSweeper(_tuning()).sweep(dataset, BlendPoint(0.5, 0.25))
        blended = next(c for c in result.cells if c.weight == 0.5)
        assert np.isfinite(blended.log_loss)
        assert result.best.weight < 1.0

    def test_a_draw_rate_of_zero_leaves_no_draw_probability(self):
        probabilities = elo_outcome_from_expected(0.6, 0.0)
        assert probabilities.draw == pytest.approx(0.0)

    def test_a_certain_expected_score_still_normalizes(self):
        probabilities = elo_outcome_from_expected(1.0, 0.25)
        total = probabilities.home_win + probabilities.draw + probabilities.away_win
        assert total == pytest.approx(1.0)

    def test_zero_bootstrap_samples_reports_no_interval(self):
        result = EuropeanBlendSweeper(_tuning(bootstrap_samples=0)).sweep(
            _dataset(), BlendPoint(0.6, 0.25)
        )
        assert result.improvement_interval == (0.0, 0.0)
        assert not result.is_conclusive

    def test_the_improvement_is_never_negative(self):
        """The best cell is chosen by log-loss, so it cannot lose to the grid.

        It can still tie the incumbent when the incumbent sits on the grid and
        wins, which is why this is >= rather than >.
        """
        result = EuropeanBlendSweeper(_tuning()).sweep(
            _dataset(), BlendPoint(0.5, 0.25)
        )
        assert result.improvement >= -1e-12

    def test_a_single_match_can_still_be_swept(self):
        result = EuropeanBlendSweeper(_tuning()).sweep(
            _dataset(n=1), BlendPoint(0.6, 0.25)
        )
        assert result.matches == 1

    def test_the_confidence_level_widens_the_interval(self):
        narrow = EuropeanBlendSweeper(
            _tuning(confidence_level=0.5, bootstrap_samples=200)
        ).sweep(_dataset(20), BlendPoint(0.2, 0.25))
        wide = EuropeanBlendSweeper(
            _tuning(confidence_level=0.99, bootstrap_samples=200)
        ).sweep(_dataset(20), BlendPoint(0.2, 0.25))
        assert (wide.improvement_interval[1] - wide.improvement_interval[0]) >= (
            narrow.improvement_interval[1] - narrow.improvement_interval[0]
        )


# ── configuration variations ─────────────────────────────────────────────


class TestConfigurationVariations:
    def test_tuning_defaults_exist_for_a_config_without_the_section(self):
        """A config predating this work must still load and predict."""
        prediction = EuropeanPredictionConfig(
            enabled=True,
            dixon_coles_weight=0.6,
            min_matches_per_team=10,
            elo_draw_rate=0.25,
        )
        assert prediction.tuning.blend_weight_grid
        assert prediction.tuning.draw_rate_grid

    def test_the_default_grids_bracket_the_current_config(self):
        """A sweep that could not reproduce 0.6 would be measuring nothing."""
        tuning = EuropeanTuningConfig()
        assert min(tuning.blend_weight_grid) == 0.0
        assert max(tuning.blend_weight_grid) == 1.0
        assert 0.6 in tuning.blend_weight_grid

    def test_a_zero_holdout_season_count_is_rejected(self):
        with pytest.raises(ValueError):
            EuropeanTuningConfig(holdout_seasons=0)

    def test_a_month_outside_the_calendar_is_rejected(self):
        with pytest.raises(ValueError):
            EuropeanTuningConfig(season_start_month=13)

    def test_a_confidence_level_of_one_is_rejected(self):
        with pytest.raises(ValueError):
            EuropeanTuningConfig(confidence_level=1.0)


class TestUnreachableWithoutForcing:
    """Two guards that real data cannot easily reach, pinned deliberately."""

    def test_a_splitter_that_yields_no_folds_ends_the_run(self):
        """Defensive: the splitter derives folds from the dates it is given,
        so an empty list means a substituted splitter, not bad data."""

        class NoFolds(SeasonSplitter):
            def folds(self, dates: pd.Series) -> list[BacktestFold]:
                return []

        backtester = EuropeanBacktester(
            corpus=CrossCompetitionCorpus(
                domestic=_domestic(), supplementary=_supplementary()
            ),
            prediction=_prediction(),
            models=BacktestModels(
                elo=lambda: FootballELO(_elo_config()), dixon_coles=StubDixonColes
            ),
            splitter=NoFolds(_tuning()),
        )
        assert backtester.run().is_empty

    def test_counting_an_empty_training_frame_gives_no_counts(self):
        assert EuropeanBacktester._match_counts(pd.DataFrame()) == {}
