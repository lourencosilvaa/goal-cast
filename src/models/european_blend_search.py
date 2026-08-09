"""Empirical selection of the European blend weight and ELO draw rate.

``dixon_coles_weight`` was set to 0.6 by analogy with the domestic blend and
``elo_draw_rate`` to 0.25 as a rough observed rate. Neither was measured. This
sweeps both against held-out matches, and — like the domestic sweep it mirrors
— is strictly report-only: it never writes configuration.

The two parameters are swept together because they interact. The draw rate
belongs to the ELO leg alone, so a badly chosen one makes ELO score worse than
it should and drags the weight toward Dixon-Coles. Varying it costs nothing:
the backtest stores raw expected scores, so a different draw rate is a
recomputation rather than a re-walk.

Two things are reported beyond the winning cell, because on roughly 900
matches the winning cell alone would overstate what was learned:

* **Baselines** — each leg alone, and a constant prediction. The useful
  question is not only which weight is best but whether blending beats either
  model on its own, or beats knowing nothing at all.
* **An interval** on the improvement over the value currently in config. A
  log-loss gap of 0.002 between neighbouring cells is noise, and presenting it
  as a result would be exactly the overstatement the model labels exist to
  prevent.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from config.config_loader import EuropeanTuningConfig
from src.models.blend_search import BlendWeightSweeper, multiclass_log_loss
from src.models.european_backtest import EuropeanBacktestSet
from src.models.european_predictor import elo_outcome_from_expected

__all__ = [
    "BlendPoint",
    "BlendCell",
    "BlendBaseline",
    "EuropeanBlendSweepResult",
    "EuropeanBlendSweeper",
    "elo_outcome_from_expected",
]

#: Number of mutually exclusive 1X2 outcomes. The class order itself
#: (away=0, draw=1, home=2) is owned by :mod:`src.models.blend_search`.
_NUM_OUTCOMES = 3


@dataclass(frozen=True)
class BlendPoint:
    """One (weight, draw rate) pair — a single argument instead of two (§4.3)."""

    weight: float
    draw_rate: float


@dataclass(frozen=True)
class BlendCell:
    """One grid cell and how it scored."""

    weight: float
    draw_rate: float
    log_loss: float
    brier: float
    accuracy: float


@dataclass(frozen=True)
class BlendBaseline:
    """A reference point the sweep result should be read against."""

    name: str
    log_loss: float
    brier: float
    accuracy: float


@dataclass(frozen=True)
class EuropeanBlendSweepResult:
    """The full grid, the winner, the baselines, and the uncertainty."""

    cells: list[BlendCell]
    best: BlendCell
    baselines: list[BlendBaseline]
    incumbent: BlendPoint
    improvement: float
    improvement_interval: tuple[float, float]
    matches: int

    @property
    def is_conclusive(self) -> bool:
        """Whether the improvement over the incumbent survives resampling.

        False is a perfectly good answer and the likely one: it means the
        measurement cannot distinguish the best cell from what is already
        configured, and the honest response is to leave the config alone.
        """
        return self.improvement_interval[0] > 0.0


class EuropeanBlendSweeper:
    """Scores every (weight, draw rate) cell against a held-out set.

    Pure: takes arrays, returns numbers. No fitting, no I/O, no configuration
    written back — which is what lets it be tested on four hand-written
    matches with no corpus in sight.
    """

    #: Names used for the reference rows, so a caller formatting a table does
    #: not have to invent them.
    ELO_ONLY: ClassVar[str] = "elo only"
    DIXON_COLES_ONLY: ClassVar[str] = "dixon-coles only"
    BASE_RATE: ClassVar[str] = "base rate"
    CURRENT: ClassVar[str] = "current config"

    def __init__(self, config: EuropeanTuningConfig) -> None:
        self.config = config

    def sweep(
        self, dataset: EuropeanBacktestSet, incumbent: BlendPoint
    ) -> EuropeanBlendSweepResult:
        if dataset.is_empty:
            raise ValueError(
                "cannot sweep: the backtest produced no matches — every "
                "held-out fixture was refused, or no fold met "
                "min_holdout_matches"
            )

        cells = self._grid(dataset)
        best = min(cells, key=lambda cell: cell.log_loss)

        best_proba = self._blend(dataset, BlendPoint(best.weight, best.draw_rate))
        incumbent_proba = self._blend(dataset, incumbent)
        improvement = self._log_loss(
            incumbent_proba, dataset.outcomes
        ) - self._log_loss(best_proba, dataset.outcomes)

        return EuropeanBlendSweepResult(
            cells=cells,
            best=best,
            baselines=self._baselines(dataset, incumbent),
            incumbent=incumbent,
            improvement=improvement,
            improvement_interval=self._interval(
                incumbent_proba, best_proba, dataset.outcomes
            ),
            matches=len(dataset),
        )

    # ── the grid ─────────────────────────────────────────────────────────

    def _grid(self, dataset: EuropeanBacktestSet) -> list[BlendCell]:
        """Every combination, scored. Ordering follows the configured grids.

        The log-loss column comes from :class:`BlendWeightSweeper` — the same
        sweeper the domestic blend uses — rather than from a second
        implementation of the same arithmetic.
        """
        cells: list[BlendCell] = []
        for draw_rate in self.config.draw_rate_grid:
            elo = self._elo_matrix(dataset.elo_expected_home, draw_rate)
            swept = BlendWeightSweeper(self.config.blend_weight_grid).sweep(
                elo, dataset.dixon_coles_proba, dataset.outcomes
            )
            losses = dict(swept.per_weight)
            for weight in self.config.blend_weight_grid:
                blended = self._mix(elo, dataset.dixon_coles_proba, weight)
                cells.append(
                    BlendCell(
                        weight=weight,
                        draw_rate=draw_rate,
                        log_loss=losses[weight],
                        brier=self._brier(blended, dataset.outcomes),
                        accuracy=self._accuracy(blended, dataset.outcomes),
                    )
                )
        return cells

    def _baselines(
        self, dataset: EuropeanBacktestSet, incumbent: BlendPoint
    ) -> list[BlendBaseline]:
        """Reference points, at the incumbent's draw rate where one applies.

        The base rate is the empirical class frequency of the *evaluation*
        outcomes, which is deliberately generous — a real deployment could not
        know it in advance. That is the point: a blend that cannot beat a
        constant fitted on the answers is not worth serving.
        """
        outcomes = dataset.outcomes
        elo = self._elo_matrix(dataset.elo_expected_home, incumbent.draw_rate)
        frequencies = np.bincount(outcomes, minlength=_NUM_OUTCOMES) / len(outcomes)
        constant = np.tile(frequencies, (len(outcomes), 1))

        references = {
            self.ELO_ONLY: elo,
            self.DIXON_COLES_ONLY: dataset.dixon_coles_proba,
            self.BASE_RATE: constant,
            self.CURRENT: self._blend(dataset, incumbent),
        }
        return [
            BlendBaseline(
                name=name,
                log_loss=self._log_loss(proba, outcomes),
                brier=self._brier(proba, outcomes),
                accuracy=self._accuracy(proba, outcomes),
            )
            for name, proba in references.items()
        ]

    # ── uncertainty ──────────────────────────────────────────────────────

    def _interval(
        self, incumbent: np.ndarray, best: np.ndarray, outcomes: np.ndarray
    ) -> tuple[float, float]:
        """Percentile interval on the per-match improvement, by bootstrap.

        Resampling matches rather than folds: fixtures are the unit the loss
        is averaged over, and there are only a handful of folds.
        """
        samples = self.config.bootstrap_samples
        if samples == 0:
            return (0.0, 0.0)

        rng = np.random.default_rng(self.config.bootstrap_seed)
        n = len(outcomes)
        differences = np.empty(samples, dtype=float)
        for i in range(samples):
            picked = rng.integers(0, n, size=n)
            differences[i] = self._log_loss(
                incumbent[picked], outcomes[picked]
            ) - self._log_loss(best[picked], outcomes[picked])

        tail = (1.0 - self.config.confidence_level) / 2.0 * 100.0
        low, high = np.percentile(differences, [tail, 100.0 - tail])
        return (float(low), float(high))

    # ── arithmetic ───────────────────────────────────────────────────────

    @staticmethod
    def _elo_matrix(expected_home: np.ndarray, draw_rate: float) -> np.ndarray:
        """The ELO leg as [away, draw, home].

        Built through :func:`elo_outcome_from_expected`, the same function the
        served predictor calls, so the tuned model cannot drift from it.
        """
        triples = [
            elo_outcome_from_expected(float(expected), draw_rate)
            for expected in expected_home
        ]
        return np.array([[t.away_win, t.draw, t.home_win] for t in triples])

    @classmethod
    def _mix(
        cls, elo: np.ndarray, dixon_coles: np.ndarray, weight: float
    ) -> np.ndarray:
        blended = (1.0 - weight) * elo + weight * dixon_coles
        # Both inputs sum to 1, so the mix does too; renormalize defensively
        # against floating-point drift.
        return np.asarray(blended / blended.sum(axis=1, keepdims=True), dtype=float)

    def _blend(self, dataset: EuropeanBacktestSet, point: BlendPoint) -> np.ndarray:
        """One blended probability matrix at an arbitrary point.

        Used for the incumbent, which need not sit on the grid — 0.6 is not on
        a 0.05 grid, and silently rounding it to the nearest cell would report
        an improvement over a value nobody configured.
        """
        elo = self._elo_matrix(dataset.elo_expected_home, point.draw_rate)
        return self._mix(elo, dataset.dixon_coles_proba, point.weight)

    @staticmethod
    def _log_loss(proba: np.ndarray, outcomes: np.ndarray) -> float:
        return multiclass_log_loss(proba, outcomes)

    @staticmethod
    def _brier(proba: np.ndarray, outcomes: np.ndarray) -> float:
        """Multiclass Brier score — the squared error of the whole triple."""
        actual = np.zeros_like(proba)
        actual[np.arange(len(outcomes)), outcomes] = 1.0
        return float(np.mean(np.sum((proba - actual) ** 2, axis=1)))

    @staticmethod
    def _accuracy(proba: np.ndarray, outcomes: np.ndarray) -> float:
        return float(np.mean(proba.argmax(axis=1) == outcomes))
