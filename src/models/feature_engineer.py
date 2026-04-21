import numpy as np
import pandas as pd

from config.config_loader import FeaturesConfig


class FeatureEngineer:
    """
    Feature generation based on historical team statistics.
    For each match, uses ONLY data available BEFORE the match starts.
    """

    def __init__(self, config: FeaturesConfig) -> None:
        self.config = config
        self.window = config.rolling_window

    def compute_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling averages for each team over the last N matches."""
        df = df.sort_values("Date").copy()

        home_records = df[
            [
                "Date",
                "HomeTeam",
                "FTHG",
                "FTAG",
                "HS",
                "AS",
                "HST",
                "AST",
                "HC",
                "AC",
                "HF",
                "AF",
            ]
        ].copy()
        home_records.columns = [
            "Date",
            "Team",
            "GF",
            "GA",
            "Shots",
            "ShotsAgainst",
            "SoT",
            "SoTAgainst",
            "Corners",
            "CornersAgainst",
            "Fouls",
            "FoulsAgainst",
        ]
        home_records["IsHome"] = 1

        away_records = df[
            [
                "Date",
                "AwayTeam",
                "FTAG",
                "FTHG",
                "AS",
                "HS",
                "AST",
                "HST",
                "AC",
                "HC",
                "AF",
                "HF",
            ]
        ].copy()
        away_records.columns = [
            "Date",
            "Team",
            "GF",
            "GA",
            "Shots",
            "ShotsAgainst",
            "SoT",
            "SoTAgainst",
            "Corners",
            "CornersAgainst",
            "Fouls",
            "FoulsAgainst",
        ]
        away_records["IsHome"] = 0

        all_records = pd.concat([home_records, away_records])
        all_records = all_records.sort_values("Date")

        stats_cols = [
            "GF",
            "GA",
            "Shots",
            "ShotsAgainst",
            "SoT",
            "SoTAgainst",
            "Corners",
            "CornersAgainst",
            "Fouls",
            "FoulsAgainst",
        ]

        rolling_stats: dict[str, pd.DataFrame] = {}
        for team in all_records["Team"].unique():
            team_data = all_records[all_records["Team"] == team].copy()
            for col in stats_cols:
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

    def build_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join home and away team statistics for each match."""
        team_stats = self.compute_team_stats(df)

        stat_features = [c for c in team_stats.columns if c.startswith("avg_")]
        stat_features.append("Form")

        features_list: list[dict[str, float | int]] = []

        for idx, match in df.iterrows():
            home = match["HomeTeam"]
            away = match["AwayTeam"]
            date = match["Date"]

            home_stats = team_stats[
                (team_stats["Team"] == home)
                & (team_stats["Date"] == date)
                & (team_stats["IsHome"] == 1)
            ]
            away_stats = team_stats[
                (team_stats["Team"] == away)
                & (team_stats["Date"] == date)
                & (team_stats["IsHome"] == 0)
            ]

            if home_stats.empty or away_stats.empty:
                continue

            row: dict[str, float | int] = {"match_idx": int(idx)}  # type: ignore[arg-type]
            for feat in stat_features:
                h_val = home_stats[feat].values[0]
                a_val = away_stats[feat].values[0]
                row[f"home_{feat}"] = h_val
                row[f"away_{feat}"] = a_val
                row[f"diff_{feat}"] = h_val - a_val

            features_list.append(row)

        features_df = pd.DataFrame(features_list).set_index("match_idx")
        result = df.join(features_df, how="inner")
        return result.dropna(subset=[c for c in features_df.columns])

    def add_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert bookmaker odds to probabilities and add as features."""
        df = df.copy()

        if all(col in df.columns for col in ["B365H", "B365D", "B365A"]):
            df["odds_prob_H"] = 1 / df["B365H"]
            df["odds_prob_D"] = 1 / df["B365D"]
            df["odds_prob_A"] = 1 / df["B365A"]

            total = df["odds_prob_H"] + df["odds_prob_D"] + df["odds_prob_A"]
            df["norm_prob_H"] = df["odds_prob_H"] / total
            df["norm_prob_D"] = df["odds_prob_D"] / total
            df["norm_prob_A"] = df["odds_prob_A"] / total

            df["odds_spread"] = df["norm_prob_H"] - df["norm_prob_A"]

        return df

    def compute_xg_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling xG proxy from PREVIOUS matches (no leakage).

        Instead of using the current match's shot stats (which are
        post-match data), we use the rolling average of xG from
        prior matches for each team.
        """
        df = df.sort_values("Date").copy()
        sot_conv = self.config.xg_proxy.sot_conversion
        shot_conv = self.config.xg_proxy.shot_conversion

        if "HST" not in df.columns or "HS" not in df.columns:
            return df

        # Compute per-match xG (used only as intermediate for rolling)
        df["_match_home_xG"] = (
            df["HST"] * sot_conv + (df["HS"] - df["HST"]).clip(lower=0) * shot_conv
        )
        df["_match_away_xG"] = (
            df["AST"] * sot_conv + (df["AS"] - df["AST"]).clip(lower=0) * shot_conv
        )

        # Build rolling xG per team from their prior matches
        window = self.window
        home_rolling_xg: list[float] = []
        away_rolling_xg: list[float] = []
        home_rolling_xga: list[float] = []
        away_rolling_xga: list[float] = []

        # Track each team's xG history
        team_xg_for: dict[str, list[float]] = {}  # goals created
        team_xg_against: dict[str, list[float]] = {}  # goals conceded

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]

            # Get rolling averages BEFORE this match
            h_for = team_xg_for.get(home, [])
            h_against = team_xg_against.get(home, [])
            a_for = team_xg_for.get(away, [])
            a_against = team_xg_against.get(away, [])

            home_rolling_xg.append(
                sum(h_for[-window:]) / len(h_for[-window:]) if h_for else 0
            )
            away_rolling_xg.append(
                sum(a_for[-window:]) / len(a_for[-window:]) if a_for else 0
            )
            home_rolling_xga.append(
                sum(h_against[-window:]) / len(h_against[-window:]) if h_against else 0
            )
            away_rolling_xga.append(
                sum(a_against[-window:]) / len(a_against[-window:]) if a_against else 0
            )

            # Update history AFTER recording (no leakage)
            team_xg_for.setdefault(home, []).append(row["_match_home_xG"])
            team_xg_against.setdefault(home, []).append(row["_match_away_xG"])
            team_xg_for.setdefault(away, []).append(row["_match_away_xG"])
            team_xg_against.setdefault(away, []).append(row["_match_home_xG"])

        df["home_xG_rolling"] = home_rolling_xg
        df["away_xG_rolling"] = away_rolling_xg
        df["home_xGA_rolling"] = home_rolling_xga
        df["away_xGA_rolling"] = away_rolling_xga
        df["xG_diff"] = df["home_xG_rolling"] - df["away_xG_rolling"]

        # Clean up intermediate columns
        df.drop(columns=["_match_home_xG", "_match_away_xG"], inplace=True)

        return df

    def compute_fatigue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rest days between matches as fatigue indicator."""
        df = df.sort_values("Date").copy()
        max_rest = self.config.fatigue.max_rest_days
        default_rest = self.config.fatigue.default_rest_days
        threshold = self.config.fatigue.fatigue_threshold

        rest_days_home: list[int] = []
        rest_days_away: list[int] = []
        last_match: dict[str, pd.Timestamp] = {}

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            date = row["Date"]

            if home in last_match:
                delta = (date - last_match[home]).days
                rest_days_home.append(min(delta, max_rest))
            else:
                rest_days_home.append(default_rest)

            if away in last_match:
                delta = (date - last_match[away]).days
                rest_days_away.append(min(delta, max_rest))
            else:
                rest_days_away.append(default_rest)

            last_match[home] = date
            last_match[away] = date

        df["home_rest_days"] = rest_days_home
        df["away_rest_days"] = rest_days_away
        df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
        df["home_fatigued"] = (df["home_rest_days"] <= threshold).astype(int)
        df["away_fatigued"] = (df["away_rest_days"] <= threshold).astype(int)
        df["is_midweek"] = df["Date"].dt.dayofweek.isin([1, 2]).astype(int)

        return df

    def compute_h2h_features(self, df: pd.DataFrame, n_last: int = 5) -> pd.DataFrame:
        """Head-to-head statistics between teams."""
        df = df.sort_values("Date").copy()
        h2h_features: list[dict[str, float]] = []

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            date = row["Date"]

            prev = df[
                (df["Date"] < date)
                & (
                    ((df["HomeTeam"] == home) & (df["AwayTeam"] == away))
                    | ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
                )
            ].tail(n_last)

            if len(prev) < 2:
                h2h_features.append(
                    {
                        "h2h_home_wins": np.nan,
                        "h2h_draws": np.nan,
                        "h2h_total_goals_avg": np.nan,
                    }
                )
                continue

            home_wins = 0
            draws = 0
            total_goals = 0

            for _, p in prev.iterrows():
                if p["HomeTeam"] == home:
                    if p["FTR"] == "H":
                        home_wins += 1
                    elif p["FTR"] == "D":
                        draws += 1
                    total_goals += p["FTHG"] + p["FTAG"]
                else:
                    if p["FTR"] == "A":
                        home_wins += 1
                    elif p["FTR"] == "D":
                        draws += 1
                    total_goals += p["FTHG"] + p["FTAG"]

            n = len(prev)
            h2h_features.append(
                {
                    "h2h_home_wins": home_wins / n,
                    "h2h_draws": draws / n,
                    "h2h_total_goals_avg": total_goals / n,
                }
            )

        h2h_df = pd.DataFrame(h2h_features, index=df.index)
        return pd.concat([df, h2h_df], axis=1)

    def compute_draw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features that specifically help predict draws.

        Draws correlate with:
        - Teams of similar strength (small ELO/form gap)
        - Low-scoring matches
        - Defensive teams (low goals conceded)
        - Historical draw tendency
        """
        df = df.sort_values("Date").copy()

        # Draw tendency: rolling percentage of draws for each team
        team_draw_pct: dict[str, list[int]] = {}
        home_draw_pct: list[float] = []
        away_draw_pct: list[float] = []

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]

            h_hist = team_draw_pct.get(home, [])
            a_hist = team_draw_pct.get(away, [])

            # Use last N matches before this one
            w = self.window
            home_draw_pct.append(sum(h_hist[-w:]) / len(h_hist[-w:]) if h_hist else 0)
            away_draw_pct.append(sum(a_hist[-w:]) / len(a_hist[-w:]) if a_hist else 0)

            # Update after recording
            is_draw = 1 if row.get("FTR") == "D" else 0
            team_draw_pct.setdefault(home, []).append(is_draw)
            team_draw_pct.setdefault(away, []).append(is_draw)

        df["home_draw_pct"] = home_draw_pct
        df["away_draw_pct"] = away_draw_pct
        df["avg_draw_pct"] = (df["home_draw_pct"] + df["away_draw_pct"]) / 2

        # Strength similarity (absolute difference in form — smaller = more likely draw)
        if "home_Form" in df.columns and "away_Form" in df.columns:
            df["form_gap"] = (df["home_Form"] - df["away_Form"]).abs()

        # Goal-scoring rate similarity
        if "home_avg_GF" in df.columns and "away_avg_GF" in df.columns:
            df["attack_similarity"] = 1 / (
                1 + (df["home_avg_GF"] - df["away_avg_GF"]).abs()
            )
            df["defense_similarity"] = 1 / (
                1 + (df["home_avg_GA"] - df["away_avg_GA"]).abs()
            )

        # Combined defensive strength (both teams concede few goals → draw likely)
        if "home_avg_GA" in df.columns:
            df["combined_defensive"] = 1 / (1 + df["home_avg_GA"]) + 1 / (
                1 + df["away_avg_GA"]
            )

        return df

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature engineering pipeline."""
        df = self.build_match_features(df)
        df = self.add_odds_features(df)
        df = self.compute_xg_proxy(df)
        df = self.compute_fatigue_features(df)
        df = self.compute_h2h_features(df)
        df = self.compute_draw_features(df)
        return df
