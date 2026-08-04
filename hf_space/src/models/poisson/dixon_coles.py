"""Dixon-Coles bivariate Poisson score model.

Goals are modelled directly: for a fixture the home side scores
``Poisson(lambda_home)`` and the away side ``Poisson(lambda_away)`` with

    lambda_home = exp(attack_home + defense_away + home_advantage)
    lambda_away = exp(attack_away + defense_home)

where ``attack`` (higher = scores more) and ``defense`` (higher = concedes
more) are per-team strengths. Dixon & Coles (1997) add a low-score
dependency term ``rho`` correcting the four scorelines 0-0, 0-1, 1-0 and 1-1,
where independence over- or under-states the true frequencies.

Parameters are fit by maximum likelihood with exponential time-decay
weighting (recent matches count more) and a small ridge penalty that keeps
the otherwise shift-invariant attack/defense strengths identifiable.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.models.outcome_model import OutcomeModel, OutcomeProbabilities

if TYPE_CHECKING:
    # Import for typing only: the pickled artifact stores primitives, so no
    # deploy target needs PoissonConfig at runtime to load or use the model.
    from config.config_loader import PoissonConfig


@dataclass
class DixonColesPrediction:
    """Full score-model output for a fixture.

    Carries the 1X2 triple, expected goals, the over/under and BTTS markets
    and the most likely scorelines — the same market set the app already
    displays, so this drops straight into the existing payload shape.
    """

    home_win: float
    draw: float
    away_win: float
    lambda_home: float
    lambda_away: float
    over_15: float
    over_25: float
    over_35: float
    under_25: float
    btts_yes: float
    btts_no: float
    top_scorelines: list[tuple[int, int, float]]

    def as_outcome(self) -> OutcomeProbabilities:
        """Return just the 1X2 triple."""
        return OutcomeProbabilities(self.home_win, self.draw, self.away_win)


class DixonColesModel(OutcomeModel):
    """Maximum-likelihood Dixon-Coles model over a pool of teams."""

    # Ridge strength keeping the shift-invariant strengths identifiable.
    _RIDGE: ClassVar[float] = 1e-3
    # Number of top scorelines exposed in a prediction.
    _TOP_SCORELINES: ClassVar[int] = 5

    def __init__(self, config: "PoissonConfig") -> None:
        # Store only primitives (not the pydantic config) so the pickled
        # artifact stays portable across deploy targets whose config module
        # may not define PoissonConfig (e.g. the HuggingFace Space).
        self.max_goals = config.max_goals
        self.half_life_days = config.half_life_days
        self.blend_weight = config.blend_weight
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_advantage: float = 0.0
        self.rho: float = 0.0

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "DixonColesModel":
        """Fit strengths from match data (HomeTeam, AwayTeam, FTHG, FTAG, Date)."""
        data = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
        index = {team: i for i, team in enumerate(teams)}
        n = len(teams)

        home_idx = data["HomeTeam"].map(index).to_numpy(dtype=int)
        away_idx = data["AwayTeam"].map(index).to_numpy(dtype=int)
        hg = data["FTHG"].to_numpy(dtype=int)
        ag = data["FTAG"].to_numpy(dtype=int)
        weights = self._time_weights(data["Date"])

        # Parameter vector: [attack(n), defense(n), home_advantage, rho].
        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25  # sensible home-advantage start

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(home_idx, away_idx, hg, ag, weights, n),
            method="L-BFGS-B",
        )
        params = result.x
        self.attack = {team: float(params[i]) for team, i in index.items()}
        self.defense = {team: float(params[n + i]) for team, i in index.items()}
        self.home_advantage = float(params[2 * n])
        self.rho = float(params[2 * n + 1])
        return self

    def _time_weights(self, dates: pd.Series) -> np.ndarray:
        """Exponential decay weights; uniform when half-life is non-positive."""
        half_life = self.half_life_days
        if half_life <= 0:
            return np.ones(len(dates), dtype=float)
        parsed = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
        reference = parsed.max()
        age_days = (reference - parsed).dt.total_seconds() / 86400.0
        age = age_days.fillna(age_days.max()).to_numpy(dtype=float)
        weights: np.ndarray = np.power(0.5, age / half_life)
        return weights

    @classmethod
    def _neg_log_likelihood(
        cls,
        params: np.ndarray,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        hg: np.ndarray,
        ag: np.ndarray,
        weights: np.ndarray,
        n: int,
    ) -> float:
        attack = params[:n]
        defense = params[n : 2 * n]
        home_adv = params[2 * n]
        rho = params[2 * n + 1]

        lam = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
        mu = np.exp(attack[away_idx] + defense[home_idx])

        tau = cls._tau(hg, ag, lam, mu, rho)
        # Poisson log-pmf without the constant log(k!) term (irrelevant to
        # the argmin). tau is clipped to stay strictly positive for log.
        log_lik = (
            np.log(np.clip(tau, 1e-10, None))
            + hg * np.log(lam)
            - lam
            + ag * np.log(mu)
            - mu
        )
        penalty = cls._RIDGE * (np.sum(attack**2) + np.sum(defense**2))
        return float(-np.sum(weights * log_lik) + penalty)

    @staticmethod
    def _tau(
        hg: np.ndarray,
        ag: np.ndarray,
        lam: np.ndarray,
        mu: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """Dixon-Coles low-score correction, elementwise."""
        tau = np.ones_like(lam, dtype=float)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        return tau

    # -------------------------------------------------------------- predict
    def _rates(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Expected goals (lambda_home, lambda_away); 0 strength if unseen."""
        atk_h = self.attack.get(home_team, 0.0)
        atk_a = self.attack.get(away_team, 0.0)
        def_h = self.defense.get(home_team, 0.0)
        def_a = self.defense.get(away_team, 0.0)
        lam = float(np.exp(atk_h + def_a + self.home_advantage))
        mu = float(np.exp(atk_a + def_h))
        return lam, mu

    def _score_matrix(self, lam: float, mu: float) -> np.ndarray:
        """Normalized P(home=i, away=j) matrix with the DC low-score fix."""
        size = self.max_goals + 1
        goals = np.arange(size)
        home_pmf = poisson.pmf(goals, lam)
        away_pmf = poisson.pmf(goals, mu)
        matrix = np.outer(home_pmf, away_pmf)

        # Apply the four low-score corrections.
        matrix[0, 0] *= 1.0 - lam * mu * self.rho
        matrix[0, 1] *= 1.0 + lam * self.rho
        matrix[1, 0] *= 1.0 + mu * self.rho
        matrix[1, 1] *= 1.0 - self.rho

        matrix = np.clip(matrix, 0.0, None)
        total = matrix.sum()
        if total <= 0:
            uniform: np.ndarray = np.full((size, size), 1.0 / (size * size))
            return uniform
        normalized: np.ndarray = matrix / total
        return normalized

    def knows(self, team: str) -> bool:
        """Whether the team was seen during fitting (has learned strengths)."""
        return team in self.attack

    def predict(self, home_team: str, away_team: str) -> DixonColesPrediction:
        """Predict 1X2, expected goals, O/U, BTTS and top scorelines."""
        lam, mu = self._rates(home_team, away_team)
        matrix = self._score_matrix(lam, mu)
        size = matrix.shape[0]

        idx = np.arange(size)
        home_grid, away_grid = np.meshgrid(idx, idx, indexing="ij")
        totals = home_grid + away_grid

        home_win = float(matrix[home_grid > away_grid].sum())
        draw = float(np.trace(matrix))
        away_win = float(matrix[home_grid < away_grid].sum())

        over_15 = float(matrix[totals >= 2].sum())
        over_25 = float(matrix[totals >= 3].sum())
        over_35 = float(matrix[totals >= 4].sum())
        under_25 = float(matrix[totals <= 2].sum())

        btts_yes = float(matrix[(home_grid >= 1) & (away_grid >= 1)].sum())

        return DixonColesPrediction(
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            lambda_home=lam,
            lambda_away=mu,
            over_15=over_15,
            over_25=over_25,
            over_35=over_35,
            under_25=under_25,
            btts_yes=btts_yes,
            btts_no=1.0 - btts_yes,
            top_scorelines=self._top_scorelines(matrix),
        )

    def _top_scorelines(self, matrix: np.ndarray) -> list[tuple[int, int, float]]:
        """Return the most likely (home, away, prob) scorelines, desc."""
        flat = matrix.ravel()
        count = min(self._TOP_SCORELINES, flat.size)
        top_flat = np.argsort(flat)[::-1][:count]
        size = matrix.shape[1]
        return [(int(f // size), int(f % size), float(flat[f])) for f in top_flat]

    def predict_outcome(self, home_team: str, away_team: str) -> OutcomeProbabilities:
        """OutcomeModel interface: 1X2 triple for the fixture."""
        return self.predict(home_team, away_team).as_outcome()
