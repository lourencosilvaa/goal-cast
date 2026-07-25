import pandas as pd

from config.config_loader import EloConfig
from src.models.elo import FootballELO


class InternationalELO(FootballELO):
    """ELO for national teams with neutral-venue awareness.

    Many international fixtures (World Cups, continental finals) are played
    on neutral ground where the nominal home side has no venue edge. The
    home-advantage term is therefore scaled by
    ``neutral_home_advantage_factor`` for matches flagged ``Neutral``:
    ``0.0`` removes it entirely, ``1.0`` keeps the full advantage.
    """

    def __init__(
        self,
        config: EloConfig,
        neutral_home_advantage_factor: float = 0.0,
    ) -> None:
        super().__init__(config)
        self.neutral_home_advantage_factor = neutral_home_advantage_factor

    def _home_advantage_for(self, neutral: bool) -> float:
        if neutral:
            return self.config.home_advantage * self.neutral_home_advantage_factor
        return float(self.config.home_advantage)

    def compute_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chronologically compute pre-match ELO features, neutral-aware.

        Mirrors the parent implementation but applies a per-match home
        advantage derived from the ``Neutral`` flag (defaulting to a normal
        home fixture when the column is absent).
        """
        df = df.sort_values("Date").copy()
        elo_features: list[dict[str, float]] = []

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            neutral = bool(row.get("Neutral", False))
            home_adv = self._home_advantage_for(neutral)

            r_home = self.get_rating(home)
            r_away = self.get_rating(away)

            e_home = self.expected_score(r_home + home_adv, r_away)

            elo_features.append(
                {
                    "elo_home": r_home,
                    "elo_away": r_away,
                    "elo_diff": r_home - r_away,
                    "elo_expected_home": e_home,
                    "elo_expected_away": 1 - e_home,
                }
            )

            if pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG")):
                self._update_with_advantage(
                    home, away, int(row["FTHG"]), int(row["FTAG"]), home_adv
                )

        return pd.concat(
            [df.reset_index(drop=True), pd.DataFrame(elo_features)],
            axis=1,
        )

    def _update_with_advantage(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        home_adv: float,
    ) -> tuple[float, float]:
        """Update ratings using a per-match (neutral-aware) home advantage."""
        r_home = self.get_rating(home) + home_adv
        r_away = self.get_rating(away)

        e_home = self.expected_score(r_home, r_away)
        e_away = 1.0 - e_home

        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals < away_goals:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        m = self.margin_multiplier(home_goals - away_goals)

        self.ratings[home] += self.config.k_factor * m * (s_home - e_home)
        self.ratings[away] += self.config.k_factor * m * (s_away - e_away)

        return self.ratings[home], self.ratings[away]
