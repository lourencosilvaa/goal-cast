"""Empirical team and head-to-head statistics computed from match history.

Where :mod:`src.analysis.match_stats` answers *"what will happen?"* with a
Poisson model, this module answers *"what has actually happened?"* — win/draw/
loss records, goal averages, recent form and past meetings, straight from the
observed results.

Everything is derived from a cleaned match frame handed in by the caller, so
the calculator performs no I/O and needs no data-loader, model or network. The
HuggingFace Space already keeps the full history in memory and injects it; the
Render backend never imports this module (it ships without pandas).

Two collaborators are optional and injected, never imported globally:

* ``leagues`` — league-code → display-name map, used to scope a query to one
  competition and to label the answer.
* ``market_model`` — anything exposing ``knows(team)`` and ``predict(home,
  away)`` (i.e. the fitted Dixon-Coles model). When it knows both sides of a
  fixture, the goal-market block is attached; otherwise it is simply absent.

Reporting horizons come from :class:`config.config_loader.InsightsConfig`:

* ``overall`` / ``home`` / ``away`` span the **whole** frame.
* ``recent`` / ``form_sequence`` / ``rates`` / ``averages`` /
  ``recent_matches`` span the last ``recent_matches`` games.
* Head-to-head **counts** use every meeting; the meeting list is capped at
  ``h2h_matches``.

Match lists and form sequences are ordered **most recent first**.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import exp, factorial
from typing import Any, ClassVar, Protocol

import pandas as pd

from config.config_loader import InsightsConfig

#: Result letters, from the subject team's point of view.
WIN, DRAW, LOSS = "W", "D", "L"
#: Venue letters, from the subject team's point of view.
HOME, AWAY = "H", "A"


def _poisson_prob(rate: float, count: int) -> float:
    """P(X = count) for a Poisson variable with the given rate."""
    if rate <= 0:
        return 1.0 if count == 0 else 0.0
    return (rate**count) * exp(-rate) / factorial(count)


class MarketModel(Protocol):
    """Structural type of the fitted score model used for goal markets."""

    def knows(self, team: str) -> bool: ...

    def predict(self, home_team: str, away_team: str) -> Any: ...


@dataclass(frozen=True)
class TeamQuery:
    """Everything needed to profile a single team."""

    league_code: str
    team: str


@dataclass(frozen=True)
class FixtureQuery:
    """Everything needed to profile a face-off between two teams."""

    league_code: str
    home_team: str
    away_team: str


@dataclass(frozen=True)
class MatchRecord:
    """One historical match, ready for display.

    ``result`` and ``venue`` are relative to whichever team the record was
    built for, and are ``None`` when there is no subject team.
    """

    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    result: str | None = None
    venue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "result": self.result,
            "venue": self.venue,
        }


@dataclass(frozen=True)
class TeamRecord:
    """Win/draw/loss and goal counters over some set of matches.

    Only the raw counters are stored; every reported figure is derived, so a
    record can never contradict itself. All ratios collapse to ``0.0`` on an
    empty record rather than dividing by zero.
    """

    #: Points awarded for a win and for a draw (competition rules, not config).
    POINTS_PER_WIN: ClassVar[int] = 3
    POINTS_PER_DRAW: ClassVar[int] = 1

    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def points(self) -> int:
        return self.wins * self.POINTS_PER_WIN + self.draws * self.POINTS_PER_DRAW

    @property
    def points_per_game(self) -> float:
        return self._per_game(self.points)

    @property
    def win_pct(self) -> float:
        return self._per_game(self.wins)

    @property
    def draw_pct(self) -> float:
        return self._per_game(self.draws)

    @property
    def loss_pct(self) -> float:
        return self._per_game(self.losses)

    @property
    def avg_goals_for(self) -> float:
        return self._per_game(self.goals_for)

    @property
    def avg_goals_against(self) -> float:
        return self._per_game(self.goals_against)

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    def _per_game(self, total: int) -> float:
        return total / self.played if self.played else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "played": self.played,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "goals_for": self.goals_for,
            "goals_against": self.goals_against,
            "points": self.points,
            "points_per_game": self.points_per_game,
            "win_pct": self.win_pct,
            "draw_pct": self.draw_pct,
            "loss_pct": self.loss_pct,
            "avg_goals_for": self.avg_goals_for,
            "avg_goals_against": self.avg_goals_against,
            "goal_difference": self.goal_difference,
        }


@dataclass(frozen=True)
class TeamRates:
    """Share of recent matches satisfying each market condition."""

    clean_sheets: float = 0.0
    failed_to_score: float = 0.0
    btts: float = 0.0
    over_2_5: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean_sheets": self.clean_sheets,
            "failed_to_score": self.failed_to_score,
            "btts": self.btts,
            "over_2_5": self.over_2_5,
        }


@dataclass(frozen=True)
class TeamAverages:
    """Per-match averages of the secondary match statistics."""

    shots: float = 0.0
    shots_on_target: float = 0.0
    corners: float = 0.0
    cards: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "shots": self.shots,
            "shots_on_target": self.shots_on_target,
            "corners": self.corners,
            "cards": self.cards,
        }


@dataclass(frozen=True)
class TeamInsights:
    """Full statistical profile of one team within one competition."""

    team: str
    league: str
    league_code: str
    overall: TeamRecord = field(default_factory=TeamRecord)
    home: TeamRecord = field(default_factory=TeamRecord)
    away: TeamRecord = field(default_factory=TeamRecord)
    recent: TeamRecord = field(default_factory=TeamRecord)
    form_sequence: list[str] = field(default_factory=list)
    rates: TeamRates = field(default_factory=TeamRates)
    averages: TeamAverages = field(default_factory=TeamAverages)
    recent_matches: list[MatchRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "league": self.league,
            "league_code": self.league_code,
            "overall": self.overall.to_dict(),
            "home": self.home.to_dict(),
            "away": self.away.to_dict(),
            "recent": self.recent.to_dict(),
            "form_sequence": list(self.form_sequence),
            "rates": self.rates.to_dict(),
            "averages": self.averages.to_dict(),
            "recent_matches": [m.to_dict() for m in self.recent_matches],
        }


@dataclass(frozen=True)
class HeadToHead:
    """Past meetings between two teams, counted from the fixture's viewpoint.

    ``home_wins`` means "meetings won by the team playing at home in the
    upcoming fixture", regardless of where the historical meeting was played;
    the per-meeting venue is preserved in ``matches``.
    """

    home_team: str
    away_team: str
    played: int = 0
    home_wins: int = 0
    draws: int = 0
    away_wins: int = 0
    home_goals: int = 0
    away_goals: int = 0
    btts_count: int = 0
    over_2_5_count: int = 0
    matches: list[MatchRecord] = field(default_factory=list)

    @property
    def avg_goals_home(self) -> float:
        return self._per_meeting(self.home_goals)

    @property
    def avg_goals_away(self) -> float:
        return self._per_meeting(self.away_goals)

    @property
    def avg_goals_total(self) -> float:
        return self._per_meeting(self.home_goals + self.away_goals)

    @property
    def btts_pct(self) -> float:
        return self._per_meeting(self.btts_count)

    @property
    def over_2_5_pct(self) -> float:
        return self._per_meeting(self.over_2_5_count)

    def _per_meeting(self, total: int) -> float:
        return total / self.played if self.played else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "played": self.played,
            "home_wins": self.home_wins,
            "draws": self.draws,
            "away_wins": self.away_wins,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "avg_goals_home": self.avg_goals_home,
            "avg_goals_away": self.avg_goals_away,
            "avg_goals_total": self.avg_goals_total,
            "btts_pct": self.btts_pct,
            "over_2_5_pct": self.over_2_5_pct,
            "matches": [m.to_dict() for m in self.matches],
        }


@dataclass(frozen=True)
class GoalMarkets:
    """Model-derived goal markets for a fixture.

    ``source`` states which model produced them — the calibrated Dixon-Coles
    artifact or the historical-rates approximation. It is part of the payload
    on purpose: silently swapping one for the other would be exactly the kind
    of hidden behaviour the project forbids.

    Values are rounded here because this block mirrors the payload shape the
    app already renders for scheduled matches (see ``MatchStats.to_dict``).
    The empirical blocks above deliberately keep full precision — rounding
    those is the UI's business.
    """

    #: Decimal places for probabilities and for goal counts respectively.
    PROBABILITY_DIGITS: ClassVar[int] = 3
    GOALS_DIGITS: ClassVar[int] = 2
    #: Provenance labels.
    SOURCE_MODEL: ClassVar[str] = "model"
    SOURCE_HISTORICAL: ClassVar[str] = "historical"

    home_xg: float
    away_xg: float
    over_15: float
    over_25: float
    over_35: float
    under_25: float
    btts_yes: float
    btts_no: float
    top_scorelines: list[tuple[int, int, float]] = field(default_factory=list)
    source: str = SOURCE_MODEL

    @property
    def total_xg(self) -> float:
        return self.home_xg + self.away_xg

    @classmethod
    def from_prediction(
        cls,
        prediction: Any,
        max_scorelines: int,
        source: str = SOURCE_MODEL,
    ) -> "GoalMarkets":
        """Adapt a Dixon-Coles style prediction into the market block."""
        return cls(
            home_xg=float(prediction.lambda_home),
            away_xg=float(prediction.lambda_away),
            over_15=float(prediction.over_15),
            over_25=float(prediction.over_25),
            over_35=float(prediction.over_35),
            under_25=float(prediction.under_25),
            btts_yes=float(prediction.btts_yes),
            btts_no=float(prediction.btts_no),
            top_scorelines=list(prediction.top_scorelines)[:max_scorelines],
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        goals = self.GOALS_DIGITS
        prob = self.PROBABILITY_DIGITS
        return {
            "expected_goals": {
                "home": round(self.home_xg, goals),
                "away": round(self.away_xg, goals),
                "total": round(self.total_xg, goals),
            },
            "over_under": {
                "over_1_5": round(self.over_15, prob),
                "over_2_5": round(self.over_25, prob),
                "over_3_5": round(self.over_35, prob),
                "under_2_5": round(self.under_25, prob),
            },
            "btts": {
                "yes": round(self.btts_yes, prob),
                "no": round(self.btts_no, prob),
            },
            "top_scorelines": [
                {"score": f"{h}-{a}", "prob": round(p, prob)}
                for h, a, p in self.top_scorelines
            ],
            "source": self.source,
        }


@dataclass(frozen=True)
class EmpiricalPrediction:
    """Dixon-Coles-shaped output produced from historical scoring rates.

    Structurally identical to ``DixonColesPrediction`` so both can feed
    :meth:`GoalMarkets.from_prediction` unchanged.
    """

    lambda_home: float
    lambda_away: float
    over_15: float
    over_25: float
    over_35: float
    under_25: float
    btts_yes: float
    btts_no: float
    top_scorelines: list[tuple[int, int, float]]


class EmpiricalPoissonModel:
    """Score model derived from the teams' own scoring and conceding rates.

    The calibrated Dixon-Coles artifact is optional — it is absent whenever the
    deployed model repo predates it. This stands in for it, using the same
    independent-Poisson approximation the app already applies to scheduled
    matches (see :class:`src.analysis.match_stats.MatchStatsCalculator`):

        home_xg = home_attack × away_defence / league_average

    It is *not* calibrated, which is why every market it produces is labelled
    ``historical``. It implements ``knows``/``predict`` so it is substitutable
    for the fitted model wherever a :class:`MarketModel` is expected.
    """

    #: Bounds keeping a freak run of results from producing absurd rates.
    MIN_EXPECTED_GOALS: ClassVar[float] = 0.2
    MAX_EXPECTED_GOALS: ClassVar[float] = 4.0
    #: Goal line behind the over/under markets.
    OVER_LINES: ClassVar[tuple[float, ...]] = (1.5, 2.5, 3.5)

    HOME_TEAM: ClassVar[str] = "HomeTeam"
    AWAY_TEAM: ClassVar[str] = "AwayTeam"
    HOME_GOALS: ClassVar[str] = "FTHG"
    AWAY_GOALS: ClassVar[str] = "FTAG"

    def __init__(
        self,
        matches: pd.DataFrame,
        league_average_goals: float,
        max_goals: int,
    ) -> None:
        self._matches = matches
        self._league_average = league_average_goals
        self._max_goals = max_goals

    def knows(self, team: str) -> bool:
        """Whether the frame holds any match for this team."""
        if self._matches.empty or self.HOME_TEAM not in self._matches.columns:
            return False
        return bool(
            (
                (self._matches[self.HOME_TEAM] == team)
                | (self._matches[self.AWAY_TEAM] == team)
            ).any()
        )

    def predict(self, home_team: str, away_team: str) -> EmpiricalPrediction:
        """Expected goals and the markets implied by two independent Poissons."""
        home_attack = self._rate(home_team, scored=True, at_home=True)
        home_defence = self._rate(home_team, scored=False, at_home=True)
        away_attack = self._rate(away_team, scored=True, at_home=False)
        away_defence = self._rate(away_team, scored=False, at_home=False)

        average = self._league_average or 1.0
        home_xg = self._clamp(home_attack * away_defence / average)
        away_xg = self._clamp(away_attack * home_defence / average)

        scorelines = self._scoreline_matrix(home_xg, away_xg)
        totals = {
            line: sum(p for h, a, p in scorelines if h + a > line)
            for line in self.OVER_LINES
        }
        btts_yes = sum(p for h, a, p in scorelines if h > 0 and a > 0)
        over_25 = totals[2.5]
        return EmpiricalPrediction(
            lambda_home=home_xg,
            lambda_away=away_xg,
            over_15=totals[1.5],
            over_25=over_25,
            over_35=totals[3.5],
            under_25=1.0 - over_25,
            btts_yes=btts_yes,
            btts_no=1.0 - btts_yes,
            top_scorelines=sorted(scorelines, key=lambda item: -item[2]),
        )

    def _clamp(self, value: float) -> float:
        return max(self.MIN_EXPECTED_GOALS, min(value, self.MAX_EXPECTED_GOALS))

    def _rate(self, team: str, scored: bool, at_home: bool) -> float:
        """Goals scored (or conceded) per match by ``team`` at one venue.

        Falls back to the venue-agnostic rate, then to the league average, so a
        team that has only ever played away still yields a usable number.
        """
        venue = self._venue_rate(team, scored=scored, at_home=at_home)
        if venue is not None:
            return venue
        overall = self._overall_rate(team, scored=scored)
        return overall if overall is not None else self._league_average

    def _venue_rate(self, team: str, scored: bool, at_home: bool) -> float | None:
        column = self.HOME_TEAM if at_home else self.AWAY_TEAM
        rows = self._matches[self._matches[column] == team]
        if rows.empty:
            return None
        goals = self.HOME_GOALS if scored == at_home else self.AWAY_GOALS
        return float(rows[goals].mean())

    def _overall_rate(self, team: str, scored: bool) -> float | None:
        home = self._matches[self._matches[self.HOME_TEAM] == team]
        away = self._matches[self._matches[self.AWAY_TEAM] == team]
        if home.empty and away.empty:
            return None
        home_goals = self.HOME_GOALS if scored else self.AWAY_GOALS
        away_goals = self.AWAY_GOALS if scored else self.HOME_GOALS
        values = list(home[home_goals]) + list(away[away_goals])
        return sum(float(v) for v in values) / len(values) if values else None

    def _scoreline_matrix(
        self, home_xg: float, away_xg: float
    ) -> list[tuple[int, int, float]]:
        return [
            (h, a, _poisson_prob(home_xg, h) * _poisson_prob(away_xg, a))
            for h in range(self._max_goals)
            for a in range(self._max_goals)
        ]


@dataclass(frozen=True)
class MatchInsights:
    """Everything known about an upcoming face-off between two teams."""

    home_team: str
    away_team: str
    league: str
    league_code: str
    head_to_head: HeadToHead
    home: TeamInsights
    away: TeamInsights
    goal_markets: GoalMarkets | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "league": self.league,
            "league_code": self.league_code,
            "head_to_head": self.head_to_head.to_dict(),
            "home": self.home.to_dict(),
            "away": self.away.to_dict(),
            "goal_markets": (
                self.goal_markets.to_dict() if self.goal_markets is not None else None
            ),
        }


class TeamInsightsCalculator:
    """Derives team and head-to-head insights from a cleaned match frame.

    The frame is expected in football-data.co.uk shape (``Date``, ``HomeTeam``,
    ``AwayTeam``, ``FTHG``, ``FTAG`` plus the optional secondary columns). Any
    missing optional column simply zeroes the average it feeds, so a partially
    populated season never breaks a response.
    """

    DATE: ClassVar[str] = "Date"
    HOME_TEAM: ClassVar[str] = "HomeTeam"
    AWAY_TEAM: ClassVar[str] = "AwayTeam"
    HOME_GOALS: ClassVar[str] = "FTHG"
    AWAY_GOALS: ClassVar[str] = "FTAG"
    LEAGUE: ClassVar[str] = "League"
    DATE_FORMAT: ClassVar[str] = "%Y-%m-%d"
    #: Goal line behind the "over" rates.
    OVER_LINE: ClassVar[float] = 2.5
    #: Scoreline grid size for the historical-rates market fallback: the
    #: probability mass beyond six goals a side is negligible.
    MAX_SCORELINE_GOALS: ClassVar[int] = 6
    #: Goals per side assumed when a competition has no usable score data.
    FALLBACK_LEAGUE_AVERAGE_GOALS: ClassVar[float] = 1.35
    #: (home column, away column) pairs summed into each per-match average.
    SHOT_COLUMNS: ClassVar[list[tuple[str, str]]] = [("HS", "AS")]
    SHOT_ON_TARGET_COLUMNS: ClassVar[list[tuple[str, str]]] = [("HST", "AST")]
    CORNER_COLUMNS: ClassVar[list[tuple[str, str]]] = [("HC", "AC")]
    CARD_COLUMNS: ClassVar[list[tuple[str, str]]] = [("HY", "AY"), ("HR", "AR")]

    def __init__(
        self,
        matches: pd.DataFrame,
        config: InsightsConfig,
        leagues: Mapping[str, str] | None = None,
        market_model: MarketModel | None = None,
    ) -> None:
        self.config = config
        self.leagues = dict(leagues or {})
        self.market_model = market_model
        self._matches = self._sorted(matches)
        self._league_cache: dict[str, pd.DataFrame] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def knows_team(self, query: TeamQuery) -> bool:
        """Whether the team appears at all in the requested competition."""
        return not self._team_frame(query.league_code, query.team).empty

    def team_insights(self, query: TeamQuery) -> TeamInsights:
        """Full statistical profile of one team."""
        frame = self._team_frame(query.league_code, query.team)
        window = frame.tail(self.config.recent_matches)
        recent_matches = self._match_records(window, query.team)
        return TeamInsights(
            team=query.team,
            league=self._league_name(query.league_code),
            league_code=query.league_code,
            overall=self._record(frame, query.team),
            home=self._record(
                self._venue_frame(frame, query.team, self.HOME_TEAM), query.team
            ),
            away=self._record(
                self._venue_frame(frame, query.team, self.AWAY_TEAM), query.team
            ),
            recent=self._record(window, query.team),
            form_sequence=[
                match.result
                for match in recent_matches[: self.config.form_sequence_length]
                if match.result is not None
            ],
            rates=self._rates(window, query.team),
            averages=self._averages(window, query.team),
            recent_matches=recent_matches,
        )

    def match_insights(self, query: FixtureQuery) -> MatchInsights:
        """Head-to-head history plus both team profiles for a face-off."""
        return MatchInsights(
            home_team=query.home_team,
            away_team=query.away_team,
            league=self._league_name(query.league_code),
            league_code=query.league_code,
            head_to_head=self._head_to_head(query),
            home=self.team_insights(
                TeamQuery(league_code=query.league_code, team=query.home_team)
            ),
            away=self.team_insights(
                TeamQuery(league_code=query.league_code, team=query.away_team)
            ),
            goal_markets=self._goal_markets(query),
        )

    # ── frame selection ───────────────────────────────────────────────────────

    def _sorted(self, matches: pd.DataFrame) -> pd.DataFrame:
        if matches.empty or self.DATE not in matches.columns:
            return matches
        return matches.sort_values(self.DATE).reset_index(drop=True)

    def _league_name(self, league_code: str) -> str:
        return self.leagues.get(league_code, league_code)

    def _league_frame(self, league_code: str) -> pd.DataFrame:
        """Matches of one competition; the whole frame when unlabelled."""
        if league_code in self._league_cache:
            return self._league_cache[league_code]
        frame = self._matches
        if not frame.empty and self.LEAGUE in frame.columns:
            frame = frame[frame[self.LEAGUE] == self._league_name(league_code)]
        self._league_cache[league_code] = frame
        return frame

    def _team_frame(self, league_code: str, team: str) -> pd.DataFrame:
        frame = self._league_frame(league_code)
        if frame.empty:
            return frame
        return frame[(frame[self.HOME_TEAM] == team) | (frame[self.AWAY_TEAM] == team)]

    def _venue_frame(
        self, frame: pd.DataFrame, team: str, venue_column: str
    ) -> pd.DataFrame:
        """Matches ``team`` played at one venue.

        An empty frame carries no columns at all, so the column check is what
        keeps an unloaded data set from raising instead of answering with a
        zeroed record.
        """
        if frame.empty or venue_column not in frame.columns:
            return frame
        return frame[frame[venue_column] == team]

    # ── per-team aggregation ──────────────────────────────────────────────────

    def _goals(self, row: pd.Series, team: str) -> tuple[int, int]:
        """Goals scored and conceded by ``team`` in this match."""
        scored = row[self.HOME_GOALS], row[self.AWAY_GOALS]
        if row[self.HOME_TEAM] != team:
            scored = scored[1], scored[0]
        return int(scored[0]), int(scored[1])

    @staticmethod
    def _result(goals_for: int, goals_against: int) -> str:
        if goals_for > goals_against:
            return WIN
        return DRAW if goals_for == goals_against else LOSS

    def _playable(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rows with both scores present — an unplayed fixture counts nowhere."""
        if frame.empty:
            return frame
        return frame.dropna(subset=[self.HOME_GOALS, self.AWAY_GOALS])

    def _record(self, frame: pd.DataFrame, team: str) -> TeamRecord:
        wins = draws = losses = goals_for = goals_against = 0
        played = 0
        for _, row in self._playable(frame).iterrows():
            scored, conceded = self._goals(row, team)
            played += 1
            goals_for += scored
            goals_against += conceded
            result = self._result(scored, conceded)
            if result == WIN:
                wins += 1
            elif result == DRAW:
                draws += 1
            else:
                losses += 1
        return TeamRecord(
            played=played,
            wins=wins,
            draws=draws,
            losses=losses,
            goals_for=goals_for,
            goals_against=goals_against,
        )

    def _rates(self, frame: pd.DataFrame, team: str) -> TeamRates:
        playable = self._playable(frame)
        total = len(playable)
        if not total:
            return TeamRates()
        clean_sheets = failed_to_score = btts = over = 0
        for _, row in playable.iterrows():
            scored, conceded = self._goals(row, team)
            if conceded == 0:
                clean_sheets += 1
            if scored == 0:
                failed_to_score += 1
            if scored > 0 and conceded > 0:
                btts += 1
            if scored + conceded > self.OVER_LINE:
                over += 1
        return TeamRates(
            clean_sheets=clean_sheets / total,
            failed_to_score=failed_to_score / total,
            btts=btts / total,
            over_2_5=over / total,
        )

    def _averages(self, frame: pd.DataFrame, team: str) -> TeamAverages:
        return TeamAverages(
            shots=self._column_average(frame, team, self.SHOT_COLUMNS),
            shots_on_target=self._column_average(
                frame, team, self.SHOT_ON_TARGET_COLUMNS
            ),
            corners=self._column_average(frame, team, self.CORNER_COLUMNS),
            cards=self._column_average(frame, team, self.CARD_COLUMNS),
        )

    def _column_average(
        self,
        frame: pd.DataFrame,
        team: str,
        column_pairs: list[tuple[str, str]],
    ) -> float:
        """Average of ``team``'s own side of each (home, away) column pair."""
        pairs = [
            (home, away)
            for home, away in column_pairs
            if home in frame.columns and away in frame.columns
        ]
        if frame.empty or not pairs:
            return 0.0
        total = 0.0
        counted = 0
        for _, row in frame.iterrows():
            is_home = row[self.HOME_TEAM] == team
            values = [row[home if is_home else away] for home, away in pairs]
            if any(pd.isna(value) for value in values):
                continue
            total += float(sum(float(value) for value in values))
            counted += 1
        return total / counted if counted else 0.0

    def _match_records(self, frame: pd.DataFrame, team: str) -> list[MatchRecord]:
        """Matches most recent first, annotated for ``team``."""
        records: list[MatchRecord] = []
        for _, row in self._playable(frame).iterrows():
            scored, conceded = self._goals(row, team)
            is_home = row[self.HOME_TEAM] == team
            records.append(
                MatchRecord(
                    date=self._date(row),
                    home_team=str(row[self.HOME_TEAM]),
                    away_team=str(row[self.AWAY_TEAM]),
                    home_goals=int(row[self.HOME_GOALS]),
                    away_goals=int(row[self.AWAY_GOALS]),
                    result=self._result(scored, conceded),
                    venue=HOME if is_home else AWAY,
                )
            )
        records.reverse()
        return records

    def _date(self, row: pd.Series) -> str:
        if self.DATE not in row.index:
            return ""
        value = row[self.DATE]
        if pd.isna(value):
            return ""
        return str(pd.Timestamp(value).strftime(self.DATE_FORMAT))

    # ── head-to-head ──────────────────────────────────────────────────────────

    def _head_to_head(self, query: FixtureQuery) -> HeadToHead:
        frame = self._team_frame(query.league_code, query.home_team)
        if not frame.empty:
            frame = frame[
                (frame[self.HOME_TEAM] == query.away_team)
                | (frame[self.AWAY_TEAM] == query.away_team)
            ]
        playable = self._playable(frame)
        wins = draws = losses = home_goals = away_goals = 0
        btts = over = played = 0
        for _, row in playable.iterrows():
            scored, conceded = self._goals(row, query.home_team)
            played += 1
            home_goals += scored
            away_goals += conceded
            result = self._result(scored, conceded)
            if result == WIN:
                wins += 1
            elif result == DRAW:
                draws += 1
            else:
                losses += 1
            if scored > 0 and conceded > 0:
                btts += 1
            if scored + conceded > self.OVER_LINE:
                over += 1
        return HeadToHead(
            home_team=query.home_team,
            away_team=query.away_team,
            played=played,
            home_wins=wins,
            draws=draws,
            away_wins=losses,
            home_goals=home_goals,
            away_goals=away_goals,
            btts_count=btts,
            over_2_5_count=over,
            matches=self._match_records(playable, query.home_team)[
                : self.config.h2h_matches
            ],
        )

    # ── model-derived markets ─────────────────────────────────────────────────

    def _goal_markets(self, query: FixtureQuery) -> GoalMarkets | None:
        """Calibrated markets when available, historical ones otherwise."""
        fitted = self._fitted_markets(query)
        return fitted if fitted is not None else self._historical_markets(query)

    def _fitted_markets(self, query: FixtureQuery) -> GoalMarkets | None:
        model = self.market_model
        if model is None:
            return None
        if not (model.knows(query.home_team) and model.knows(query.away_team)):
            return None
        prediction = model.predict(query.home_team, query.away_team)
        if prediction is None:
            return None
        return GoalMarkets.from_prediction(
            prediction, self.config.max_scorelines, source=GoalMarkets.SOURCE_MODEL
        )

    def _historical_markets(self, query: FixtureQuery) -> GoalMarkets | None:
        """Approximate the markets from what both teams have actually done."""
        frame = self._league_frame(query.league_code)
        model = EmpiricalPoissonModel(
            matches=frame,
            league_average_goals=self._league_average_goals(frame),
            max_goals=self.MAX_SCORELINE_GOALS,
        )
        if not (model.knows(query.home_team) and model.knows(query.away_team)):
            return None
        return GoalMarkets.from_prediction(
            model.predict(query.home_team, query.away_team),
            self.config.max_scorelines,
            source=GoalMarkets.SOURCE_HISTORICAL,
        )

    def _league_average_goals(self, frame: pd.DataFrame) -> float:
        """Mean goals per side in the competition, for the xG normaliser."""
        if frame.empty or self.HOME_GOALS not in frame.columns:
            return self.FALLBACK_LEAGUE_AVERAGE_GOALS
        home_mean = frame[self.HOME_GOALS].mean()
        away_mean = frame[self.AWAY_GOALS].mean()
        if pd.isna(home_mean) or pd.isna(away_mean):
            return self.FALLBACK_LEAGUE_AVERAGE_GOALS
        return float((home_mean + away_mean) / 2)
