from pathlib import Path
from typing import Optional

import pandas as pd

from config.config_loader import Config
from src.models.data_cleaner import DataCleaner
from src.models.international.data_loader import InternationalDataLoader
from src.models.international.elo import InternationalELO
from src.models.international.feature_engineer import InternationalFeatureEngineer
from src.models.predictor import MatchPrediction, MatchPredictor


class InternationalMatchPredictor:
    """Predicts national-team fixtures using the international model.

    Rebuilds the goals-only featured history once, then assembles a single
    feature row for a requested fixture — respecting the home/away swap rule
    (a team's own stats live under ``home_*`` or ``away_*`` depending on which
    side they played last) and recomputing ELO expectations for the requested
    neutral-venue flag.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._featured: Optional[pd.DataFrame] = None
        self._elo: Optional[InternationalELO] = None
        self._predictor: Optional[MatchPredictor] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def prepare(self) -> "InternationalMatchPredictor":
        intl = self.config.international
        loader = InternationalDataLoader(intl)
        raw = loader.load_all()
        if raw.empty:
            raise ValueError("No international data available to build features.")

        clean = DataCleaner().clean(raw)
        engineer = InternationalFeatureEngineer(self.config.features)
        featured = engineer.build_all_features(clean)

        elo = InternationalELO(
            self.config.features.elo,
            neutral_home_advantage_factor=intl.neutral_home_advantage_factor,
        )
        featured = elo.compute_elo_features(featured)

        self._featured = featured
        self._elo = elo
        self._predictor = MatchPredictor(intl.models_dir)
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _team_val(
        row: pd.Series, in_expected_role: bool, expected: str, opposite: str
    ) -> float:
        val = row.get(expected if in_expected_role else opposite, 0)
        return float(val) if pd.notna(val) else 0.0

    def _last_row(self, team: str) -> Optional[pd.Series]:
        assert self._featured is not None
        rows = self._featured[
            (self._featured["HomeTeam"] == team) | (self._featured["AwayTeam"] == team)
        ]
        return None if rows.empty else rows.iloc[-1]

    def _h2h_value(self, feat: str, home: str, away: str) -> float:
        assert self._featured is not None
        df = self._featured
        h2h_rows = df[
            ((df["HomeTeam"] == home) & (df["AwayTeam"] == away))
            | ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
        ]
        if h2h_rows.empty:
            return 0.0
        last = h2h_rows.iloc[-1]
        was_home = last.get("HomeTeam") == home
        val = last.get(feat, 0)
        val = 0.0 if pd.isna(val) else float(val)
        if feat == "h2h_home_wins" and not was_home:
            draws = last.get("h2h_draws", 0)
            draws = 0.0 if pd.isna(draws) else float(draws)
            return max(0.0, 1 - val - draws)
        return val

    def _elo_expectations(
        self, home: str, away: str, neutral: bool
    ) -> dict[str, float]:
        assert self._elo is not None
        r_home = self._elo.get_rating(home)
        r_away = self._elo.get_rating(away)
        home_adv = self._elo._home_advantage_for(neutral)
        e_home = self._elo.expected_score(r_home + home_adv, r_away)
        return {
            "elo_home": r_home,
            "elo_away": r_away,
            "elo_diff": r_home - r_away,
            "elo_expected_home": e_home,
            "elo_expected_away": 1 - e_home,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, home: str, away: str, neutral: bool = False) -> MatchPrediction:
        if self._predictor is None:
            self.prepare()
        assert self._predictor is not None

        last_home = self._last_row(home)
        last_away = self._last_row(away)
        if last_home is None or last_away is None:
            raise ValueError(
                f"Team not found in international history: "
                f"{home if last_home is None else away}"
            )

        home_was_home = last_home.get("HomeTeam") == home
        away_was_away = last_away.get("AwayTeam") == away
        elo_vals = self._elo_expectations(home, away, neutral)

        feature_row: dict[str, object] = {}
        for feat in self._predictor.feature_names:
            if feat in elo_vals:
                feature_row[feat] = elo_vals[feat]
            elif feat == "is_neutral":
                feature_row[feat] = int(neutral)
            elif feat.startswith("h2h_"):
                feature_row[feat] = self._h2h_value(feat, home, away)
            elif feat.startswith("diff_"):
                suffix = feat[len("diff_") :]
                h_val = self._team_val(
                    last_home, home_was_home, f"home_{suffix}", f"away_{suffix}"
                )
                a_val = self._team_val(
                    last_away, away_was_away, f"away_{suffix}", f"home_{suffix}"
                )
                feature_row[feat] = h_val - a_val
            elif feat.startswith("home_"):
                suffix = feat[len("home_") :]
                feature_row[feat] = self._team_val(
                    last_home, home_was_home, f"home_{suffix}", f"away_{suffix}"
                )
            elif feat.startswith("away_"):
                suffix = feat[len("away_") :]
                feature_row[feat] = self._team_val(
                    last_away, away_was_away, f"away_{suffix}", f"home_{suffix}"
                )
            elif feat == "form_gap":
                h_form = self._team_val(
                    last_home, home_was_home, "home_Form", "away_Form"
                )
                a_form = self._team_val(
                    last_away, away_was_away, "away_Form", "home_Form"
                )
                feature_row[feat] = abs(h_form - a_form)
            elif feat == "avg_draw_pct":
                h_dp = self._team_val(
                    last_home, home_was_home, "home_draw_pct", "away_draw_pct"
                )
                a_dp = self._team_val(
                    last_away, away_was_away, "away_draw_pct", "home_draw_pct"
                )
                feature_row[feat] = (h_dp + a_dp) / 2
            elif feat == "attack_similarity":
                h_gf = self._team_val(
                    last_home, home_was_home, "home_avg_GF", "away_avg_GF"
                )
                a_gf = self._team_val(
                    last_away, away_was_away, "away_avg_GF", "home_avg_GF"
                )
                feature_row[feat] = 1 / (1 + abs(h_gf - a_gf))
            elif feat == "defense_similarity":
                h_ga = self._team_val(
                    last_home, home_was_home, "home_avg_GA", "away_avg_GA"
                )
                a_ga = self._team_val(
                    last_away, away_was_away, "away_avg_GA", "home_avg_GA"
                )
                feature_row[feat] = 1 / (1 + abs(h_ga - a_ga))
            elif feat == "combined_defensive":
                h_ga = self._team_val(
                    last_home, home_was_home, "home_avg_GA", "away_avg_GA"
                )
                a_ga = self._team_val(
                    last_away, away_was_away, "away_avg_GA", "home_avg_GA"
                )
                feature_row[feat] = 1 / (1 + h_ga) + 1 / (1 + a_ga)
            else:
                val = last_home.get(feat, 0)
                feature_row[feat] = 0 if pd.isna(val) else val

        feature_row["HomeTeam"] = home
        feature_row["AwayTeam"] = away

        predictions = self._predictor.predict(pd.DataFrame([feature_row]))
        if not predictions:
            raise ValueError("Predictor returned no prediction.")
        return predictions[0]


def _models_available(models_dir: str | Path) -> bool:
    return (Path(models_dir) / "ensemble_model.joblib").exists()
