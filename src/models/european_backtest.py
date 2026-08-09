"""Rolling-origin backtest of the two models that predict European ties.

``european.prediction.dixon_coles_weight`` was set to 0.6 by analogy with the
domestic blend and never measured. Measuring it needs held-out European
matches scored by each leg separately; this module produces exactly that, and
nothing else — no blending, no scoring, no configuration written back.

Why rolling-origin rather than the single chronological split the domestic
tuner uses: only about half the corpus has both teams in a league this project
tracks, leaving roughly 190 evaluable matches in a recent season. That cannot
separate a weight of 0.5 from 0.6. Pooling several seasons, each predicted by
a model fitted only on what preceded it, is both larger and closer to how the
system actually runs — it refits periodically and serves the fixtures that
follow.

The two models need different treatment:

* **ELO** needs no per-fold work. ``compute_elo_features`` records each match's
  rating *before* applying it, so a single walk over the whole corpus yields a
  leak-free expectation for every European match in every fold at once.
* **Dixon-Coles** has no such ordering — a fit is a snapshot — so it is
  refitted per fold on matches strictly before that fold's cut.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, ClassVar

import numpy as np
import pandas as pd

from config.config_loader import EuropeanPredictionConfig, EuropeanTuningConfig
from src.models.cross_competition import (
    CrossCompetitionCorpus,
    CrossCompetitionEloBuilder,
)
from src.models.elo import FootballELO
from src.models.european_predictor import EuropeanMatchPredictor

#: Outcome encoding, matching the probability column order used by
#: ``BlendWeightSweeper`` and by every 1X2 array in this project.
AWAY_WIN = 0
DRAW = 1
HOME_WIN = 2


@dataclass(frozen=True)
class BacktestFold:
    """One held-out season and the window it covers.

    ``cut`` is both the end of the training period and the start of the
    holdout, so a match belongs to exactly one side of it.
    """

    season_start_year: int
    cut: pd.Timestamp
    end: pd.Timestamp
    evaluated: int = 0
    refused: int = 0

    @property
    def label(self) -> str:
        """The season in the form clubs and fixtures lists use: ``2024-25``."""
        return f"{self.season_start_year}-{(self.season_start_year + 1) % 100:02d}"


@dataclass(frozen=True)
class BacktestModels:
    """Factories for the two models a fold needs.

    Factories rather than instances because every fold requires a *fresh*
    Dixon-Coles: reusing one would carry the later season's fit backwards.
    Injected rather than constructed here so tests can substitute stubs and
    run without artefacts or an 18-second fit.
    """

    elo: Callable[[], FootballELO]
    dixon_coles: Callable[[], Any]


@dataclass(frozen=True)
class EuropeanBacktestSet:
    """Held-out European matches, scored by each leg separately.

    The ELO side is kept as the raw expected score rather than a 1X2 triple
    because turning it into one requires a draw rate, and the draw rate is
    itself being swept. Storing the expectation lets the sweep vary it for
    free instead of re-walking the ratings.
    """

    elo_expected_home: np.ndarray
    dixon_coles_proba: np.ndarray
    outcomes: np.ndarray
    fixtures: pd.DataFrame
    folds: list[BacktestFold]
    refused: int

    #: Columns every fixture frame carries, so an empty result has the same
    #: shape as a full one and callers need no special case.
    FIXTURE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "Date",
        "HomeTeam",
        "AwayTeam",
        "fold",
    )

    def __len__(self) -> int:
        return len(self.outcomes)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @classmethod
    def empty(cls) -> "EuropeanBacktestSet":
        return cls(
            elo_expected_home=np.array([], dtype=float),
            dixon_coles_proba=np.empty((0, 3), dtype=float),
            outcomes=np.array([], dtype=int),
            fixtures=pd.DataFrame(columns=list(cls.FIXTURE_COLUMNS)),
            folds=[],
            refused=0,
        )


@dataclass
class _FoldResult:
    """What one fold contributed, before the folds are pooled.

    Internal: a named carrier rather than a five-element tuple, so the pooling
    step reads as what it is.
    """

    fold: BacktestFold
    fixtures: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=list(EuropeanBacktestSet.FIXTURE_COLUMNS)
        )
    )
    elo_expected_home: list[float] = field(default_factory=list)
    dixon_coles_proba: list[list[float]] = field(default_factory=list)
    outcomes: list[int] = field(default_factory=list)


class SeasonSplitter:
    """Turns the corpus's dates into rolling-origin folds, one per season.

    A European season straddles the new year, so the calendar year of a match
    does not identify it: a December 2024 tie and a February 2025 tie belong
    to the same 2024-25 season. ``season_start_month`` is what settles that,
    and it is configuration rather than a literal because the answer is a
    convention, not a fact.
    """

    def __init__(self, config: EuropeanTuningConfig) -> None:
        self.config = config

    def folds(self, dates: pd.Series) -> list[BacktestFold]:
        """The most recent seasons present in ``dates``, oldest first."""
        parsed = pd.to_datetime(pd.Series(dates)).dropna()
        if parsed.empty:
            return []
        years = sorted({self._season_start_year(date) for date in parsed})
        return [self._fold(year) for year in years[-self.config.holdout_seasons :]]

    def _season_start_year(self, date: pd.Timestamp) -> int:
        if date.month >= self.config.season_start_month:
            return int(date.year)
        return int(date.year) - 1

    def _fold(self, season_start_year: int) -> BacktestFold:
        cut = pd.Timestamp(
            year=season_start_year, month=self.config.season_start_month, day=1
        )
        return BacktestFold(
            season_start_year=season_start_year,
            cut=cut,
            end=cut + pd.DateOffset(years=1),
        )


class EuropeanBacktester:
    """Produces the evaluation set the blend sweep scores.

    Owns no I/O: the corpus arrives already loaded and the models arrive as
    factories, so this runs identically on twelve fixture matches and on the
    real 60,000.
    """

    def __init__(
        self,
        corpus: CrossCompetitionCorpus,
        prediction: EuropeanPredictionConfig,
        models: BacktestModels,
        splitter: SeasonSplitter | None = None,
    ) -> None:
        self._corpus = corpus
        self._prediction = prediction
        self._models = models
        self._splitter = splitter or SeasonSplitter(prediction.tuning)

    def run(self, max_folds: int | None = None) -> EuropeanBacktestSet:
        """Walk the folds and collect every match the predictor would serve."""
        if max_folds is not None and max_folds < 1:
            raise ValueError("max_folds must be at least 1")

        european = self._walk_ratings()
        if european.empty:
            return EuropeanBacktestSet.empty()

        folds = self._splitter.folds(european["Date"])
        if max_folds is not None:
            folds = folds[-max_folds:]
        if not folds:
            return EuropeanBacktestSet.empty()

        combined = self._corpus.combined_goals()
        collected = [self._evaluate(fold, european, combined) for fold in folds]
        return self._pool(collected)

    # ── the shared ELO walk ──────────────────────────────────────────────

    def _walk_ratings(self) -> pd.DataFrame:
        """European rows carrying the ratings each side took into the tie.

        One walk covers every fold. The ratings are recorded pre-match, so a
        row's features depend only on matches that preceded it — which is what
        makes a single pass leak-free across all folds at once.
        """
        if not self._corpus.has_supplementary or self._corpus.domestic.empty:
            return pd.DataFrame()
        builder = CrossCompetitionEloBuilder(self._models.elo())
        return builder.build_all(self._corpus).supplementary

    # ── one fold ─────────────────────────────────────────────────────────

    def _evaluate(
        self, fold: BacktestFold, european: pd.DataFrame, combined: pd.DataFrame
    ) -> "_FoldResult":
        train = combined[combined["Date"] < fold.cut]
        holdout = european[
            (european["Date"] >= fold.cut) & (european["Date"] < fold.end)
        ]

        model = self._models.dixon_coles().fit(train)
        # The gates are the predictor's own, not a reimplementation of them:
        # tuning on matches the system would refuse would optimise for
        # fixtures it never serves. Counts come from ``train`` so the holdout
        # season cannot vouch for its own teams.
        predictor = EuropeanMatchPredictor(
            dixon_coles=model,
            elo=self._models.elo(),
            config=self._prediction,
            match_counts=self._match_counts(train),
        )

        rows: list[dict[str, Any]] = []
        expectations: list[float] = []
        poisson: list[list[float]] = []
        outcomes: list[int] = []
        refused = 0
        for match in holdout.to_dict("records"):
            home, away = str(match["HomeTeam"]), str(match["AwayTeam"])
            if not predictor.can_predict(home, away):
                refused += 1
                continue
            probabilities = model.predict_outcome(home, away).normalized()
            rows.append(
                {
                    "Date": match["Date"],
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "fold": fold.label,
                }
            )
            expectations.append(float(match["elo_expected_home"]))
            poisson.append(
                [probabilities.away_win, probabilities.draw, probabilities.home_win]
            )
            outcomes.append(self._encode(int(match["FTHG"]), int(match["FTAG"])))

        if len(rows) < self._prediction.tuning.min_holdout_matches:
            # Too thin to contribute anything but noise. Reported, not hidden.
            return _FoldResult(fold=replace(fold, evaluated=0, refused=refused))

        return _FoldResult(
            fold=replace(fold, evaluated=len(rows), refused=refused),
            fixtures=pd.DataFrame(
                rows, columns=list(EuropeanBacktestSet.FIXTURE_COLUMNS)
            ),
            elo_expected_home=expectations,
            dixon_coles_proba=poisson,
            outcomes=outcomes,
        )

    @staticmethod
    def _encode(home_goals: int, away_goals: int) -> int:
        if home_goals > away_goals:
            return HOME_WIN
        if home_goals < away_goals:
            return AWAY_WIN
        return DRAW

    @staticmethod
    def _match_counts(frame: pd.DataFrame) -> dict[str, int]:
        """Appearances per team, home or away, as of the fold's cut."""
        if frame.empty:
            return {}
        appearances = pd.concat(
            [frame["HomeTeam"], frame["AwayTeam"]], ignore_index=True
        )
        return {str(team): int(n) for team, n in appearances.value_counts().items()}

    # ── pooling ──────────────────────────────────────────────────────────

    def _pool(self, collected: list["_FoldResult"]) -> EuropeanBacktestSet:
        folds = [result.fold for result in collected]
        frames = [result.fixtures for result in collected if not result.fixtures.empty]
        expectations = [v for r in collected for v in r.elo_expected_home]
        poisson = [row for r in collected for row in r.dixon_coles_proba]
        outcomes = [v for r in collected for v in r.outcomes]

        if not outcomes:
            return replace(
                EuropeanBacktestSet.empty(),
                folds=folds,
                refused=sum(fold.refused for fold in folds),
            )

        return EuropeanBacktestSet(
            elo_expected_home=np.array(expectations, dtype=float),
            dixon_coles_proba=np.array(poisson, dtype=float),
            outcomes=np.array(outcomes, dtype=int),
            fixtures=pd.concat(frames, ignore_index=True),
            folds=folds,
            refused=sum(fold.refused for fold in folds),
        )
