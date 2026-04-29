import numpy as np
import pandas as pd

from config.config_loader import EloConfig


class FootballELO:
    """
    ELO ratings for football teams.

    FIFA formula: R_new = R_old + K * M * (S - E)
    where:
      K - match significance coefficient
      M - goal difference multiplier
      S - actual result (1 / 0.5 / 0)
      E - expected result by ELO
    """

    def __init__(self, config: EloConfig) -> None:
        self.config = config
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        """Get current ELO rating for a team, initializing if needed."""
        return self.ratings.setdefault(team, self.config.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probability of A beating B using the ELO formula."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def margin_multiplier(self, goal_diff: int) -> float:
        """Goal difference multiplier. Bigger wins have more impact."""
        if goal_diff == 0:
            return 1.0
        return float(np.log(abs(goal_diff) + 1))

    def update(
        self, home: str, away: str, home_goals: int, away_goals: int
    ) -> tuple[float, float]:
        """Update ratings after a match. Returns (new_home, new_away)."""
        r_home = self.get_rating(home) + self.config.home_advantage
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

    def compute_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate through all matches chronologically,
        updating ELO after each. Saves ratings BEFORE each match.
        """
        df = df.sort_values("Date").copy()
        elo_features: list[dict[str, float]] = []

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]

            r_home = self.get_rating(home)
            r_away = self.get_rating(away)

            e_home = self.expected_score(r_home + self.config.home_advantage, r_away)

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
                self.update(home, away, int(row["FTHG"]), int(row["FTAG"]))

        return pd.concat(
            [df.reset_index(drop=True), pd.DataFrame(elo_features)],
            axis=1,
        )

    def get_top_teams(self, n: int = 10) -> list[tuple[str, float]]:
        """Return top N teams by ELO rating."""
        return sorted(self.ratings.items(), key=lambda x: -x[1])[:n]
