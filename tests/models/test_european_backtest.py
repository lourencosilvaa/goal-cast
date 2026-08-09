"""Rolling-origin backtest over the European corpus.

``european.prediction.dixon_coles_weight`` was set to 0.6 by analogy with the
domestic blend and never measured. Measuring it needs held-out European
matches scored by each leg separately, which is what this builds.

Two properties matter more than the arithmetic:

* **No leakage.** Dixon-Coles is refitted per fold and must never see a match
  on or after that fold's cut date. ELO needs no such guard — it records each
  match's rating *before* applying it — but the fold's Dixon-Coles fit is a
  single snapshot and would happily train on its own test set.
* **The refusal gates are mirrored.** Only 1314 of the 2636 corpus matches
  have both teams in a tracked league. Tuning on matches the predictor would
  refuse would optimise for fixtures the system never serves.
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
    EuropeanBacktester,
    SeasonSplitter,
)
from src.models.outcome_model import OutcomeProbabilities

# ── stub models ──────────────────────────────────────────────────────────
#
# The real Dixon-Coles takes ~18s to fit on the full corpus. These tests are
# about fold construction and gating, not about goal modelling, so a stub that
# records what it was fitted on says more and runs instantly.


class StubDixonColes:
    """Records its training frame; answers a fixed distribution."""

    def __init__(self, home_win: float = 0.5, draw: float = 0.3) -> None:
        self.fitted: pd.DataFrame | None = None
        self._home_win = home_win
        self._draw = draw

    def fit(self, df: pd.DataFrame) -> "StubDixonColes":
        self.fitted = df.copy()
        return self

    def knows(self, team: str) -> bool:
        if self.fitted is None:
            return False
        seen = set(self.fitted["HomeTeam"]) | set(self.fitted["AwayTeam"])
        return team in seen

    def predict_outcome(self, home_team: str, away_team: str) -> OutcomeProbabilities:
        return OutcomeProbabilities(
            home_win=self._home_win,
            draw=self._draw,
            away_win=1.0 - self._home_win - self._draw,
        ).normalized()


class RecordingFactory:
    """Hands out stubs and keeps every one it made, for inspection."""

    def __init__(self) -> None:
        self.made: list[StubDixonColes] = []

    def __call__(self) -> StubDixonColes:
        model = StubDixonColes()
        self.made.append(model)
        return model


# ── fixtures ─────────────────────────────────────────────────────────────


def _elo_config() -> EloConfig:
    return EloConfig(k_factor=32, home_advantage=65, initial_rating=1500.0)


def _tuning(**overrides: Any) -> EuropeanTuningConfig:
    """Explicit settings — never the defaults (§7.3)."""
    settings: dict[str, Any] = {
        "blend_weight_grid": [0.0, 0.5, 1.0],
        "draw_rate_grid": [0.25],
        "holdout_seasons": 2,
        "season_start_month": 7,
        "min_holdout_matches": 1,
        "bootstrap_samples": 50,
        "bootstrap_seed": 7,
        "confidence_level": 0.95,
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
    """Two leagues, three seasons, playing every October."""
    rows = []
    for year in (2022, 2023, 2024):
        rows.extend(
            [
                ("E0", f"{year}-10-01", "Arsenal", "Everton", 3, 0),
                ("E0", f"{year}-10-08", "Everton", "Arsenal", 0, 2),
                ("P1", f"{year}-10-02", "Benfica", "Boavista", 2, 0),
                ("P1", f"{year}-10-09", "Boavista", "Benfica", 1, 1),
            ]
        )
    frame = pd.DataFrame(
        rows, columns=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    )
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["League"] = frame["Div"]
    frame["HS"] = 12
    return frame


def _supplementary() -> pd.DataFrame:
    """One cross-league tie per season, in December — inside the season."""
    rows = [
        ("CL", "2022-12-01", "Benfica", "Arsenal", 0, 4),
        ("CL", "2023-12-01", "Arsenal", "Benfica", 2, 2),
        ("CL", "2024-12-01", "Benfica", "Arsenal", 0, 3),
    ]
    frame = pd.DataFrame(
        rows, columns=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    )
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


def _backtester(
    supplementary: pd.DataFrame | None = None,
    prediction: EuropeanPredictionConfig | None = None,
    factory: RecordingFactory | None = None,
) -> EuropeanBacktester:
    corpus = CrossCompetitionCorpus(
        domestic=_domestic(),
        supplementary=_supplementary() if supplementary is None else supplementary,
    )
    return EuropeanBacktester(
        corpus=corpus,
        prediction=prediction or _prediction(),
        models=BacktestModels(
            elo=lambda: FootballELO(_elo_config()),
            dixon_coles=factory or RecordingFactory(),
        ),
    )


# ── the splitter ─────────────────────────────────────────────────────────


class TestSeasonSplitter:
    """Folds are seasons, derived from the corpus's own dates."""

    def _dates(self) -> pd.Series:
        return _supplementary()["Date"]

    def test_one_fold_per_requested_holdout_season(self):
        folds = SeasonSplitter(_tuning(holdout_seasons=2)).folds(self._dates())
        assert len(folds) == 2

    def test_folds_are_the_most_recent_seasons(self):
        folds = SeasonSplitter(_tuning(holdout_seasons=2)).folds(self._dates())
        assert [f.season_start_year for f in folds] == [2023, 2024]

    def test_folds_are_chronological(self):
        folds = SeasonSplitter(_tuning(holdout_seasons=3)).folds(self._dates())
        assert [f.cut for f in folds] == sorted(f.cut for f in folds)

    def test_the_cut_is_the_start_of_the_held_out_season(self):
        folds = SeasonSplitter(_tuning(holdout_seasons=1)).folds(self._dates())
        assert folds[0].cut == pd.Timestamp("2024-07-01")

    def test_the_season_start_month_moves_the_cut(self):
        """A December match belongs to a season that started earlier that year."""
        folds = SeasonSplitter(
            _tuning(holdout_seasons=1, season_start_month=8)
        ).folds(self._dates())
        assert folds[0].cut == pd.Timestamp("2024-08-01")

    def test_a_match_before_the_start_month_belongs_to_the_previous_season(self):
        """February 2025 is the 2024-25 season, not 2025-26."""
        dates = pd.Series(pd.to_datetime(["2025-02-01"]))
        folds = SeasonSplitter(_tuning(holdout_seasons=1)).folds(dates)
        assert folds[0].season_start_year == 2024

    def test_the_fold_ends_a_year_after_it_starts(self):
        folds = SeasonSplitter(_tuning(holdout_seasons=1)).folds(self._dates())
        assert folds[0].end == pd.Timestamp("2025-07-01")

    def test_labels_name_the_season(self):
        folds = SeasonSplitter(_tuning(holdout_seasons=1)).folds(self._dates())
        assert folds[0].label == "2024-25"

    def test_asking_for_more_seasons_than_exist_returns_what_there_is(self):
        folds = SeasonSplitter(_tuning(holdout_seasons=99)).folds(self._dates())
        assert len(folds) == 3

    def test_no_dates_yields_no_folds(self):
        folds = SeasonSplitter(_tuning()).folds(pd.Series([], dtype="datetime64[ns]"))
        assert folds == []


# ── leakage ──────────────────────────────────────────────────────────────


class TestNoLeakage:
    """The single property that decides whether the measurement means anything."""

    def test_each_fold_fits_only_on_matches_before_its_cut(self):
        factory = RecordingFactory()
        _backtester(factory=factory).run()
        folds = SeasonSplitter(_tuning()).folds(_supplementary()["Date"])
        assert len(factory.made) == len(folds)
        for model, fold in zip(factory.made, folds):
            assert model.fitted is not None
            assert model.fitted["Date"].max() < fold.cut

    def test_the_held_out_season_is_never_in_its_own_training_frame(self):
        """Stated per fold: a later fold legitimately trains on an earlier
        fold's holdout, which is what rolling-origin means."""
        factory = RecordingFactory()
        result = _backtester(factory=factory).run()
        for model, fold in zip(factory.made, result.folds):
            assert model.fitted is not None
            held_out = result.fixtures[result.fixtures["fold"] == fold.label]["Date"]
            assert not set(model.fitted["Date"]) & set(held_out)

    def test_a_fresh_model_is_fitted_for_every_fold(self):
        """State carried between folds would leak the later season backwards."""
        factory = RecordingFactory()
        _backtester(factory=factory).run()
        assert len(set(id(m) for m in factory.made)) == len(factory.made)

    def test_training_frames_grow_with_each_fold(self):
        factory = RecordingFactory()
        _backtester(factory=factory).run()
        sizes = [len(m.fitted) for m in factory.made if m.fitted is not None]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_the_training_frame_carries_both_corpora(self):
        """Cross-league identification is the whole reason the corpus exists."""
        factory = RecordingFactory()
        _backtester(factory=factory).run()
        fitted = factory.made[-1].fitted
        assert fitted is not None
        pairs = set(zip(fitted["HomeTeam"], fitted["AwayTeam"]))
        assert ("Benfica", "Arsenal") in pairs  # European
        assert ("Arsenal", "Everton") in pairs  # domestic


# ── the evaluation set ───────────────────────────────────────────────────


class TestBacktestSet:
    def test_it_evaluates_the_held_out_european_matches(self):
        result = _backtester().run()
        assert len(result) == 2  # two holdout seasons, one tie each

    def test_domestic_matches_are_not_evaluated(self):
        result = _backtester().run()
        pairs = set(zip(result.fixtures["HomeTeam"], result.fixtures["AwayTeam"]))
        assert ("Arsenal", "Everton") not in pairs

    def test_outcomes_are_encoded_away_draw_home(self):
        """0=away, 1=draw, 2=home — the order BlendWeightSweeper expects."""
        result = _backtester().run()
        by_date = dict(zip(result.fixtures["Date"], result.outcomes))
        assert by_date[pd.Timestamp("2023-12-01")] == 1  # 2-2 draw
        assert by_date[pd.Timestamp("2024-12-01")] == 0  # Benfica 0-3 Arsenal

    def test_elo_expectations_are_probabilities(self):
        result = _backtester().run()
        assert np.all((result.elo_expected_home > 0) & (result.elo_expected_home < 1))

    def test_dixon_coles_rows_are_distributions(self):
        result = _backtester().run()
        assert np.allclose(result.dixon_coles_proba.sum(axis=1), 1.0)

    def test_every_array_aligns_with_the_fixture_frame(self):
        result = _backtester().run()
        assert (
            len(result.elo_expected_home)
            == len(result.dixon_coles_proba)
            == len(result.outcomes)
            == len(result.fixtures)
        )

    def test_fixtures_record_which_fold_they_came_from(self):
        result = _backtester().run()
        assert set(result.fixtures["fold"]) == {"2023-24", "2024-25"}

    def test_elo_expectations_are_pre_match(self):
        """The tie is scored on prior form, not on its own result.

        Arsenal beats Everton twice a season and Benfica only manages a draw
        with Boavista, so by the 2024-25 tie Arsenal is the stronger side and
        Benfica — at home, with the home advantage — is still below even.
        """
        result = _backtester().run()
        last = result.fixtures["Date"] == pd.Timestamp("2024-12-01")
        assert float(result.elo_expected_home[last.to_numpy()][0]) < 0.5


# ── the gates ────────────────────────────────────────────────────────────


class TestRefusalGatesAreMirrored:
    """Evaluate only what the predictor would actually serve."""

    def _with_untracked_team(self) -> pd.DataFrame:
        extra = pd.DataFrame(
            {
                "Div": ["CL"],
                "Date": pd.to_datetime(["2024-12-02"]),
                "HomeTeam": ["FC Kairat"],
                "AwayTeam": ["Arsenal"],
                "FTHG": [0],
                "FTAG": [4],
            }
        )
        return pd.concat([_supplementary(), extra], ignore_index=True)

    def test_a_match_with_an_untracked_team_is_not_evaluated(self):
        result = _backtester(supplementary=self._with_untracked_team()).run()
        assert "FC Kairat" not in set(result.fixtures["HomeTeam"])

    def test_refusals_are_counted_not_silently_dropped(self):
        result = _backtester(supplementary=self._with_untracked_team()).run()
        assert result.refused == 1

    def test_the_thin_evidence_gate_is_applied(self):
        """Same threshold the predictor uses, measured as of the fold's cut."""
        strict = _prediction(min_matches_per_team=1000)
        result = _backtester(prediction=strict).run()
        assert len(result) == 0
        assert result.refused == 2

    def test_match_counts_are_taken_as_of_the_cut(self):
        """Counting the whole corpus would let the holdout season vouch for itself.

        Each team plays two domestic matches and one European tie a season, so
        by the 2023-24 cut it has three to its name and by the 2024-25 cut six.
        A threshold of five must therefore admit the later fold only.
        """
        result = _backtester(prediction=_prediction(min_matches_per_team=5)).run()
        assert set(result.fixtures["fold"]) == {"2024-25"}


# ── boundaries ───────────────────────────────────────────────────────────


class TestBoundaries:
    def test_a_fold_below_the_minimum_is_skipped(self):
        prediction = _prediction(tuning=_tuning(min_holdout_matches=2))
        result = _backtester(prediction=prediction).run()
        assert len(result) == 0

    def test_max_folds_caps_the_work(self):
        result = _backtester().run(max_folds=1)
        assert set(result.fixtures["fold"]) == {"2024-25"}

    def test_max_folds_keeps_the_most_recent_seasons(self):
        result = _backtester().run(max_folds=1)
        assert result.folds[0].label == "2024-25"

    def test_an_empty_corpus_yields_an_empty_result(self):
        result = _backtester(supplementary=pd.DataFrame()).run()
        assert result.is_empty

    def test_an_empty_result_is_reported_not_crashed(self):
        result = _backtester(supplementary=pd.DataFrame()).run()
        assert len(result) == 0
        assert result.folds == []

    def test_each_fold_reports_what_it_evaluated(self):
        result = _backtester().run()
        assert [f.evaluated for f in result.folds] == [1, 1]

    def test_zero_max_folds_is_rejected(self):
        with pytest.raises(ValueError):
            _backtester().run(max_folds=0)
