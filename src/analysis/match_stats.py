"""
Extended match statistics using Poisson model.

Computes goal-based predictions from historical team averages:
- Expected goals per team
- Over/Under 2.5 probability
- Both Teams To Score (BTTS)
- Most likely scorelines
"""

from dataclasses import dataclass, field
from math import exp, factorial

import pandas as pd

from config.config_loader import DataConfig
from src.models.poisson.dixon_coles import DixonColesModel


@dataclass
class TeamStats:
    """Historical averages for a team."""

    team: str
    avg_goals_scored_home: float = 0.0
    avg_goals_conceded_home: float = 0.0
    avg_goals_scored_away: float = 0.0
    avg_goals_conceded_away: float = 0.0
    avg_shots: float = 0.0
    avg_shots_on_target: float = 0.0
    avg_corners: float = 0.0
    matches_played: int = 0
    clean_sheets_pct: float = 0.0
    btts_pct: float = 0.0
    over25_pct: float = 0.0
    form_points: float = 0.0  # avg points last N matches


@dataclass
class MatchStats:
    """Extended match statistics from Poisson model."""

    home_team: str
    away_team: str
    league: str
    # Expected goals
    home_xg: float = 0.0
    away_xg: float = 0.0
    total_xg: float = 0.0
    # Over/Under
    over15_prob: float = 0.0
    over25_prob: float = 0.0
    over35_prob: float = 0.0
    under25_prob: float = 0.0
    # BTTS
    btts_yes_prob: float = 0.0
    btts_no_prob: float = 0.0
    # Top scorelines (home_goals, away_goals, probability)
    top_scorelines: list[tuple[int, int, float]] = field(default_factory=list)
    # Team form
    home_form: float = 0.0
    away_form: float = 0.0
    # Historical BTTS/O2.5 percentages
    home_btts_pct: float = 0.0
    away_btts_pct: float = 0.0
    home_over25_pct: float = 0.0
    away_over25_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "league": self.league,
            "expected_goals": {
                "home": round(self.home_xg, 2),
                "away": round(self.away_xg, 2),
                "total": round(self.total_xg, 2),
            },
            "over_under": {
                "over_1.5": round(self.over15_prob, 3),
                "over_2.5": round(self.over25_prob, 3),
                "over_3.5": round(self.over35_prob, 3),
                "under_2.5": round(self.under25_prob, 3),
            },
            "btts": {
                "yes": round(self.btts_yes_prob, 3),
                "no": round(self.btts_no_prob, 3),
            },
            "top_scorelines": [
                {"score": f"{h}-{a}", "prob": round(p, 3)}
                for h, a, p in self.top_scorelines[:5]
            ],
            "form": {
                "home": round(self.home_form, 2),
                "away": round(self.away_form, 2),
            },
        }


def _poisson_prob(lam: float, k: int) -> float:
    """P(X = k) for Poisson distribution with rate lambda."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam**k) * exp(-lam) / factorial(k)


def _score_prob(home_lambda: float, away_lambda: float, h: int, a: int) -> float:
    """Probability of a specific scoreline."""
    return _poisson_prob(home_lambda, h) * _poisson_prob(away_lambda, a)


class MatchStatsCalculator:
    """Compute extended match statistics from historical data."""

    def __init__(
        self,
        data_config: DataConfig,
        n_recent: int = 10,
        poisson_model: DixonColesModel | None = None,
    ) -> None:
        self.data_config = data_config
        self.n_recent = n_recent
        # When a fitted Dixon-Coles model is supplied and it knows both
        # teams, the goal-based markets are derived from its calibrated
        # scoreline distribution instead of the naive independent Poisson.
        self.poisson_model = poisson_model
        self._data_cache: dict[str, pd.DataFrame] = {}

    def _load_league_data(self, league_code: str) -> pd.DataFrame:
        """Load historical data for a league (cached)."""
        if league_code in self._data_cache:
            return self._data_cache[league_code]

        from src.models.data_loader import FootballDataLoader

        loader = FootballDataLoader(self.data_config)
        frames = []
        for season in self.data_config.seasons:
            df = loader.load_season(league_code, season)
            if not df.empty:
                frames.append(df)

        if not frames:
            self._data_cache[league_code] = pd.DataFrame()
            return self._data_cache[league_code]

        data = pd.concat(frames, ignore_index=True)

        # Parse dates
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
            try:
                data["Date"] = pd.to_datetime(data["Date"], format=fmt)
                break
            except (ValueError, TypeError):
                continue

        data = data.sort_values("Date").reset_index(drop=True)
        self._data_cache[league_code] = data
        return data

    def compute_team_stats(self, league_code: str, team: str) -> TeamStats:
        """Compute historical averages for a team."""
        data = self._load_league_data(league_code)
        if data.empty:
            return TeamStats(team=team)

        # Home matches
        home = data[data["HomeTeam"] == team].tail(self.n_recent * 2)
        # Away matches
        away = data[data["AwayTeam"] == team].tail(self.n_recent * 2)

        # All recent matches (home + away) sorted by date
        team_matches = data[
            (data["HomeTeam"] == team) | (data["AwayTeam"] == team)
        ].tail(self.n_recent)

        if home.empty and away.empty:
            return TeamStats(team=team)

        home_recent = home.tail(self.n_recent)
        away_recent = away.tail(self.n_recent)

        stats = TeamStats(team=team)
        stats.matches_played = len(team_matches)

        if not home_recent.empty and "FTHG" in home_recent.columns:
            stats.avg_goals_scored_home = home_recent["FTHG"].mean()
            stats.avg_goals_conceded_home = home_recent["FTAG"].mean()

        if not away_recent.empty and "FTAG" in away_recent.columns:
            stats.avg_goals_scored_away = away_recent["FTAG"].mean()
            stats.avg_goals_conceded_away = away_recent["FTHG"].mean()

        # Shots stats (from all recent)
        if not team_matches.empty:
            home_m = team_matches[team_matches["HomeTeam"] == team]
            away_m = team_matches[team_matches["AwayTeam"] == team]
            shots = []
            sot = []
            corners = []
            if "HS" in data.columns:
                shots.extend(home_m["HS"].dropna().tolist())
                shots.extend(away_m["AS"].dropna().tolist())
                sot.extend(home_m["HST"].dropna().tolist())
                sot.extend(away_m["AST"].dropna().tolist())
            if "HC" in data.columns:
                corners.extend(home_m["HC"].dropna().tolist())
                corners.extend(away_m["AC"].dropna().tolist())
            stats.avg_shots = sum(shots) / len(shots) if shots else 0
            stats.avg_shots_on_target = sum(sot) / len(sot) if sot else 0
            stats.avg_corners = sum(corners) / len(corners) if corners else 0

        # Clean sheets, BTTS, Over 2.5 from recent matches
        if not team_matches.empty and "FTHG" in team_matches.columns:
            clean_sheets = 0
            btts_count = 0
            over25_count = 0
            form_pts = 0

            for _, row in team_matches.iterrows():
                is_home = row["HomeTeam"] == team
                gf = row["FTHG"] if is_home else row["FTAG"]
                ga = row["FTAG"] if is_home else row["FTHG"]
                total = row["FTHG"] + row["FTAG"]

                if ga == 0:
                    clean_sheets += 1
                if row["FTHG"] > 0 and row["FTAG"] > 0:
                    btts_count += 1
                if total > 2.5:
                    over25_count += 1

                # Points
                if gf > ga:
                    form_pts += 3
                elif gf == ga:
                    form_pts += 1

            n = len(team_matches)
            stats.clean_sheets_pct = clean_sheets / n
            stats.btts_pct = btts_count / n
            stats.over25_pct = over25_count / n
            stats.form_points = form_pts / n

        return stats

    def _dixon_coles_match_stats(
        self,
        league_code: str,
        home_team: str,
        away_team: str,
        home_stats: TeamStats,
        away_stats: TeamStats,
    ) -> MatchStats:
        """Build MatchStats from the fitted Dixon-Coles scoreline model."""
        assert self.poisson_model is not None
        dc = self.poisson_model.predict(home_team, away_team)
        return MatchStats(
            home_team=home_team,
            away_team=away_team,
            league=self.data_config.leagues.get(league_code, league_code),
            home_xg=dc.lambda_home,
            away_xg=dc.lambda_away,
            total_xg=dc.lambda_home + dc.lambda_away,
            over15_prob=dc.over_15,
            over25_prob=dc.over_25,
            over35_prob=dc.over_35,
            under25_prob=dc.under_25,
            btts_yes_prob=dc.btts_yes,
            btts_no_prob=dc.btts_no,
            top_scorelines=dc.top_scorelines,
            home_form=home_stats.form_points,
            away_form=away_stats.form_points,
            home_btts_pct=home_stats.btts_pct,
            away_btts_pct=away_stats.btts_pct,
            home_over25_pct=home_stats.over25_pct,
            away_over25_pct=away_stats.over25_pct,
        )

    def compute_match_stats(
        self,
        league_code: str,
        home_team: str,
        away_team: str,
    ) -> MatchStats:
        """Compute extended Poisson-based match statistics."""
        data = self._load_league_data(league_code)
        home_stats = self.compute_team_stats(league_code, home_team)
        away_stats = self.compute_team_stats(league_code, away_team)

        # Prefer the calibrated Dixon-Coles markets when the model knows both
        # teams; the historical form/percentage fields still come from data.
        if (
            self.poisson_model is not None
            and self.poisson_model.knows(home_team)
            and self.poisson_model.knows(away_team)
        ):
            return self._dixon_coles_match_stats(
                league_code, home_team, away_team, home_stats, away_stats
            )

        # League average goals per game
        if not data.empty and "FTHG" in data.columns:
            league_avg = (data["FTHG"].mean() + data["FTAG"].mean()) / 2
        else:
            league_avg = 1.35  # fallback

        # Poisson expected goals
        # Home xG = home_attack * away_defense / league_avg * home_advantage
        home_attack = (
            home_stats.avg_goals_scored_home
            if home_stats.avg_goals_scored_home > 0
            else league_avg
        )
        away_defense = (
            away_stats.avg_goals_conceded_away
            if away_stats.avg_goals_conceded_away > 0
            else league_avg
        )
        away_attack = (
            away_stats.avg_goals_scored_away
            if away_stats.avg_goals_scored_away > 0
            else league_avg
        )
        home_defense = (
            home_stats.avg_goals_conceded_home
            if home_stats.avg_goals_conceded_home > 0
            else league_avg
        )

        if league_avg > 0:
            home_xg = home_attack * away_defense / league_avg
            away_xg = away_attack * home_defense / league_avg
        else:
            home_xg = 1.5
            away_xg = 1.0

        # Clamp to reasonable range
        home_xg = max(0.3, min(home_xg, 4.0))
        away_xg = max(0.2, min(away_xg, 3.5))

        # Score probability matrix (0-5 goals each)
        max_goals = 6
        score_probs: list[tuple[int, int, float]] = []
        for h in range(max_goals):
            for a in range(max_goals):
                p = _score_prob(home_xg, away_xg, h, a)
                score_probs.append((h, a, p))

        score_probs.sort(key=lambda x: -x[2])

        # Over/Under
        under15 = sum(p for h, a, p in score_probs if h + a <= 1)
        under25 = sum(p for h, a, p in score_probs if h + a <= 2)
        under35 = sum(p for h, a, p in score_probs if h + a <= 3)

        # BTTS
        btts_no = sum(p for h, a, p in score_probs if h == 0 or a == 0)

        return MatchStats(
            home_team=home_team,
            away_team=away_team,
            league=self.data_config.leagues.get(league_code, league_code),
            home_xg=home_xg,
            away_xg=away_xg,
            total_xg=home_xg + away_xg,
            over15_prob=1 - under15,
            over25_prob=1 - under25,
            over35_prob=1 - under35,
            under25_prob=under25,
            btts_yes_prob=1 - btts_no,
            btts_no_prob=btts_no,
            top_scorelines=score_probs[:8],
            home_form=home_stats.form_points,
            away_form=away_stats.form_points,
            home_btts_pct=home_stats.btts_pct,
            away_btts_pct=away_stats.btts_pct,
            home_over25_pct=home_stats.over25_pct,
            away_over25_pct=away_stats.over25_pct,
        )
