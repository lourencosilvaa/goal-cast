import pandas as pd

from config.config_loader import FeaturesConfig
from src.models.feature_engineer import FeatureEngineer
from src.models.international.base import AbstractFeatureEngineer


class InternationalFeatureEngineer(FeatureEngineer, AbstractFeatureEngineer):
    """Goals-only feature engineering for national-team matches.

    The Kaggle international dataset carries only goals (no shots, corners,
    fouls, cards or odds), so this engineer overrides team-stat computation
    to roll goals scored/conceded and form, then reuses the parent's generic
    match join, head-to-head and draw features. Bookmaker-odds, xG-proxy and
    fatigue steps are intentionally skipped. A ``is_neutral`` flag is added as
    a direct model signal for neutral-venue fixtures.
    """

    _GOAL_STAT_COLS = ["GF", "GA"]

    def compute_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Roll goals-for/against and form per team (goals-only source)."""
        df = df.sort_values("Date").copy()

        home_records = df[["Date", "HomeTeam", "FTHG", "FTAG"]].copy()
        home_records.columns = ["Date", "Team", "GF", "GA"]
        home_records["IsHome"] = 1

        away_records = df[["Date", "AwayTeam", "FTAG", "FTHG"]].copy()
        away_records.columns = ["Date", "Team", "GF", "GA"]
        away_records["IsHome"] = 0

        all_records = pd.concat([home_records, away_records]).sort_values("Date")

        rolling_stats: dict[str, pd.DataFrame] = {}
        for team in all_records["Team"].unique():
            team_data = all_records[all_records["Team"] == team].copy()
            for col in self._GOAL_STAT_COLS:
                team_data[f"avg_{col}"] = (
                    team_data[col]
                    .shift(1)
                    .rolling(window=self.window, min_periods=3)
                    .mean()
                )
            team_data["Points"] = team_data.apply(
                lambda r: 3 if r["GF"] > r["GA"] else (1 if r["GF"] == r["GA"] else 0),
                axis=1,
            )
            team_data["Form"] = (
                team_data["Points"]
                .shift(1)
                .rolling(window=self.window, min_periods=3)
                .mean()
            )
            rolling_stats[team] = team_data

        return pd.concat(rolling_stats.values())

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the goals-only feature pipeline (no odds/xG/fatigue steps)."""
        df = self.build_match_features(df)
        df = self.compute_h2h_features(df)
        df = self.compute_draw_features(df)
        if "Neutral" in df.columns:
            df["is_neutral"] = df["Neutral"].astype(int)
        return df

    def __init__(self, config: FeaturesConfig) -> None:
        super().__init__(config)
