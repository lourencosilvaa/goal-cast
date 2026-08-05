"""Tests for the empirical team / head-to-head insight calculator.

Every test builds its own deterministic match frame and passes an explicit
``InsightsConfig`` — nothing here depends on the shipped YAML defaults or on
any network/disk access (§7.3).

Semantics under test (documented once here, asserted throughout):

* ``overall`` / ``home`` / ``away`` cover the **full** history in the frame.
* ``recent`` / ``form_sequence`` / ``rates`` / ``averages`` / ``recent_matches``
  cover only the last ``recent_matches`` games.
* Head-to-head **counts** cover every meeting; the ``matches`` list is capped
  at ``h2h_matches``.
* Match lists and form sequences are ordered **most recent first**.
"""

import pandas as pd
import pytest

from config.config_loader import InsightsConfig
from src.analysis.team_insights import (
    FixtureQuery,
    GoalMarkets,
    MatchRecord,
    TeamInsightsCalculator,
    TeamQuery,
    TeamRecord,
)

LEAGUES = {"P1": "Liga Portugal", "E0": "Premier League"}

# Sporting's five Liga Portugal matches, chronologically:
#   1  H  Sporting 2-1 Porto    → W
#   2  A  Benfica  0-0 Sporting → D
#   3  H  Sporting 3-1 Benfica  → W
#   4  A  Porto    2-0 Sporting → L
#   5  H  Sporting 1-1 Porto    → D
_ROWS = [
    # date, home, away, fthg, ftag, hs, as_, hst, ast, hc, ac, hy, ay, hr, ar, league
    ("2024-01-01", "Sporting", "Porto", 2, 1, 10, 9, 5, 4, 6, 3, 1, 2, 0, 0, "P1"),
    ("2024-01-08", "Benfica", "Sporting", 0, 0, 11, 8, 4, 3, 5, 4, 1, 2, 0, 0, "P1"),
    ("2024-01-15", "Sporting", "Benfica", 3, 1, 14, 7, 7, 2, 8, 3, 1, 1, 0, 0, "P1"),
    ("2024-01-22", "Porto", "Sporting", 2, 0, 13, 6, 6, 2, 7, 2, 2, 3, 0, 1, "P1"),
    ("2024-01-29", "Sporting", "Porto", 1, 1, 12, 10, 3, 5, 5, 6, 2, 1, 0, 0, "P1"),
    ("2024-02-05", "Porto", "Benfica", 1, 0, 9, 8, 3, 3, 4, 5, 1, 2, 0, 0, "P1"),
    # Same club name in a different league — must never leak into P1 answers.
    ("2024-01-10", "Sporting", "Arsenal", 5, 0, 20, 4, 9, 1, 9, 1, 0, 3, 0, 0, "E0"),
]

_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
    "LeagueCode",
]


def matches_frame() -> pd.DataFrame:
    """Cleaned match frame in the shape the Space keeps in memory."""
    df = pd.DataFrame(_ROWS, columns=_COLUMNS)
    df["Date"] = pd.to_datetime(df["Date"])
    df["League"] = df["LeagueCode"].map(LEAGUES)
    return df.drop(columns=["LeagueCode"])


def wide_config() -> InsightsConfig:
    """Window wide enough to span the whole fixture set."""
    return InsightsConfig(
        recent_matches=10, h2h_matches=10, form_sequence_length=10, max_scorelines=5
    )


def narrow_config() -> InsightsConfig:
    """Window deliberately smaller than the history, to prove truncation."""
    return InsightsConfig(
        recent_matches=3, h2h_matches=2, form_sequence_length=3, max_scorelines=3
    )


def calculator(config: InsightsConfig, market_model=None) -> TeamInsightsCalculator:
    return TeamInsightsCalculator(
        matches=matches_frame(),
        config=config,
        leagues=LEAGUES,
        market_model=market_model,
    )


class StubMarketModel:
    """Duck-typed stand-in for the fitted Dixon-Coles model."""

    def __init__(self, known: set[str], prediction=None) -> None:
        self._known = known
        self._prediction = prediction

    def knows(self, team: str) -> bool:
        return team in self._known

    def predict(self, home_team: str, away_team: str):
        return self._prediction


class StubPrediction:
    """Field-for-field stand-in for ``DixonColesPrediction``."""

    lambda_home = 1.8
    lambda_away = 1.1
    over_15 = 0.78
    over_25 = 0.55
    over_35 = 0.31
    under_25 = 0.45
    btts_yes = 0.52
    btts_no = 0.48
    top_scorelines = [
        (2, 1, 0.11),
        (1, 1, 0.10),
        (1, 0, 0.09),
        (2, 0, 0.08),
        (3, 1, 0.05),
    ]


class TestTeamRecord:
    """The W/D/L value object derives everything from its raw counters."""

    def test_points_and_points_per_game(self):
        record = TeamRecord(
            played=5, wins=2, draws=2, losses=1, goals_for=6, goals_against=5
        )
        assert record.points == 8
        assert record.points_per_game == pytest.approx(1.6)

    def test_result_percentages(self):
        record = TeamRecord(
            played=4, wins=2, draws=1, losses=1, goals_for=5, goals_against=3
        )
        assert record.win_pct == pytest.approx(0.5)
        assert record.draw_pct == pytest.approx(0.25)
        assert record.loss_pct == pytest.approx(0.25)

    def test_goal_averages_and_difference(self):
        record = TeamRecord(
            played=4, wins=2, draws=1, losses=1, goals_for=6, goals_against=2
        )
        assert record.avg_goals_for == pytest.approx(1.5)
        assert record.avg_goals_against == pytest.approx(0.5)
        assert record.goal_difference == 4

    def test_zero_played_never_divides_by_zero(self):
        record = TeamRecord(
            played=0, wins=0, draws=0, losses=0, goals_for=0, goals_against=0
        )
        assert record.points_per_game == 0.0
        assert record.win_pct == 0.0
        assert record.avg_goals_for == 0.0
        assert record.avg_goals_against == 0.0

    def test_to_dict_exposes_derived_fields(self):
        record = TeamRecord(
            played=5, wins=2, draws=2, losses=1, goals_for=6, goals_against=5
        )
        payload = record.to_dict()
        assert payload["played"] == 5
        assert payload["wins"] == 2
        assert payload["draws"] == 2
        assert payload["losses"] == 1
        assert payload["goals_for"] == 6
        assert payload["goals_against"] == 5
        assert payload["points"] == 8
        assert payload["points_per_game"] == pytest.approx(1.6)


class TestMatchRecord:

    def test_to_dict_serialises_date_as_iso(self):
        record = MatchRecord(
            date="2024-01-01",
            home_team="Sporting",
            away_team="Porto",
            home_goals=2,
            away_goals=1,
            result="W",
            venue="H",
        )
        assert record.to_dict() == {
            "date": "2024-01-01",
            "home_team": "Sporting",
            "away_team": "Porto",
            "home_goals": 2,
            "away_goals": 1,
            "result": "W",
            "venue": "H",
        }


class TestTeamInsightsFullHistory:
    """``overall``/``home``/``away`` span every match in the frame."""

    @staticmethod
    def _sporting():
        return calculator(wide_config()).team_insights(
            TeamQuery(league_code="P1", team="Sporting")
        )

    def test_identifies_team_and_league(self):
        insights = self._sporting()
        assert insights.team == "Sporting"
        assert insights.league == "Liga Portugal"
        assert insights.league_code == "P1"

    def test_overall_record_counts_both_venues(self):
        overall = self._sporting().overall
        assert (overall.played, overall.wins, overall.draws, overall.losses) == (
            5,
            2,
            2,
            1,
        )
        assert overall.goals_for == 6
        assert overall.goals_against == 5

    def test_home_record_covers_only_home_matches(self):
        home = self._sporting().home
        assert (home.played, home.wins, home.draws, home.losses) == (3, 2, 1, 0)
        assert home.goals_for == 6
        assert home.goals_against == 3

    def test_away_record_covers_only_away_matches(self):
        away = self._sporting().away
        assert (away.played, away.wins, away.draws, away.losses) == (2, 0, 1, 1)
        assert away.goals_for == 0
        assert away.goals_against == 2

    def test_other_leagues_are_excluded(self):
        """The E0 'Sporting' 5-0 must not inflate the Liga Portugal record."""
        overall = self._sporting().overall
        assert overall.played == 5
        assert overall.goals_for == 6

    def test_league_scoped_query_reads_the_right_row(self):
        insights = calculator(wide_config()).team_insights(
            TeamQuery(league_code="E0", team="Sporting")
        )
        assert insights.league == "Premier League"
        assert insights.overall.played == 1
        assert insights.overall.goals_for == 5


class TestTeamInsightsRecentWindow:
    """``recent``/``rates``/``form_sequence`` honour ``recent_matches``."""

    @staticmethod
    def _sporting(config):
        return calculator(config).team_insights(
            TeamQuery(league_code="P1", team="Sporting")
        )

    def test_recent_record_is_truncated_to_the_window(self):
        recent = self._sporting(narrow_config()).recent
        assert (recent.played, recent.wins, recent.draws, recent.losses) == (3, 1, 1, 1)
        assert recent.goals_for == 4
        assert recent.goals_against == 4

    def test_form_sequence_is_most_recent_first(self):
        assert self._sporting(narrow_config()).form_sequence == ["D", "L", "W"]

    def test_form_sequence_capped_independently_of_the_window(self):
        config = InsightsConfig(
            recent_matches=10, h2h_matches=10, form_sequence_length=2, max_scorelines=5
        )
        assert self._sporting(config).form_sequence == ["D", "L"]

    def test_rates_computed_over_the_window(self):
        rates = self._sporting(narrow_config()).rates
        assert rates.clean_sheets == pytest.approx(0.0)
        assert rates.failed_to_score == pytest.approx(1 / 3)
        assert rates.btts == pytest.approx(2 / 3)
        assert rates.over_2_5 == pytest.approx(1 / 3)

    def test_rates_over_full_history(self):
        rates = self._sporting(wide_config()).rates
        assert rates.clean_sheets == pytest.approx(0.2)
        assert rates.failed_to_score == pytest.approx(0.4)
        assert rates.btts == pytest.approx(0.6)
        assert rates.over_2_5 == pytest.approx(0.4)

    def test_averages_blend_home_and_away_columns(self):
        averages = self._sporting(wide_config()).averages
        assert averages.shots == pytest.approx(10.0)
        assert averages.shots_on_target == pytest.approx(4.0)
        assert averages.corners == pytest.approx(5.0)
        assert averages.cards == pytest.approx(2.0)

    def test_recent_matches_are_most_recent_first(self):
        recent = self._sporting(narrow_config()).recent_matches
        assert [m.date for m in recent] == ["2024-01-29", "2024-01-22", "2024-01-15"]

    def test_recent_matches_carry_result_and_venue_for_the_subject_team(self):
        recent = self._sporting(narrow_config()).recent_matches
        assert [(m.result, m.venue) for m in recent] == [
            ("D", "H"),
            ("L", "A"),
            ("W", "H"),
        ]

    def test_recent_match_scores_keep_home_away_orientation(self):
        loss = self._sporting(narrow_config()).recent_matches[1]
        assert (loss.home_team, loss.away_team) == ("Porto", "Sporting")
        assert (loss.home_goals, loss.away_goals) == (2, 0)


class TestTeamInsightsSerialisation:

    def test_to_dict_shape(self):
        payload = (
            calculator(wide_config())
            .team_insights(TeamQuery(league_code="P1", team="Sporting"))
            .to_dict()
        )
        assert payload["team"] == "Sporting"
        assert payload["league"] == "Liga Portugal"
        assert payload["league_code"] == "P1"
        assert payload["overall"]["played"] == 5
        assert payload["home"]["wins"] == 2
        assert payload["away"]["losses"] == 1
        assert payload["recent"]["played"] == 5
        assert payload["form_sequence"] == ["D", "L", "W", "D", "W"]
        assert payload["rates"]["btts"] == pytest.approx(0.6)
        assert payload["averages"]["shots"] == pytest.approx(10.0)
        assert len(payload["recent_matches"]) == 5
        assert payload["recent_matches"][0]["date"] == "2024-01-29"


class TestHeadToHead:

    @staticmethod
    def _h2h(config):
        return calculator(config).match_insights(
            FixtureQuery(league_code="P1", home_team="Sporting", away_team="Porto")
        ).head_to_head

    def test_counts_cover_every_meeting(self):
        h2h = self._h2h(narrow_config())
        assert h2h.played == 3
        assert h2h.home_wins == 1
        assert h2h.draws == 1
        assert h2h.away_wins == 1

    def test_goals_are_attributed_per_named_team(self):
        h2h = self._h2h(wide_config())
        assert h2h.home_goals == 3
        assert h2h.away_goals == 4

    def test_goal_averages(self):
        h2h = self._h2h(wide_config())
        assert h2h.avg_goals_home == pytest.approx(1.0)
        assert h2h.avg_goals_away == pytest.approx(4 / 3)
        assert h2h.avg_goals_total == pytest.approx(7 / 3)

    def test_market_rates(self):
        h2h = self._h2h(wide_config())
        assert h2h.btts_pct == pytest.approx(2 / 3)
        assert h2h.over_2_5_pct == pytest.approx(1 / 3)

    def test_match_list_is_capped_and_ordered_most_recent_first(self):
        h2h = self._h2h(narrow_config())
        assert [m.date for m in h2h.matches] == ["2024-01-29", "2024-01-22"]

    def test_match_results_are_from_the_home_teams_perspective(self):
        h2h = self._h2h(wide_config())
        assert [m.result for m in h2h.matches] == ["D", "L", "W"]

    def test_teams_that_never_met_yield_an_empty_record(self):
        h2h = calculator(wide_config()).match_insights(
            FixtureQuery(league_code="P1", home_team="Sporting", away_team="Nobody")
        ).head_to_head
        assert h2h.played == 0
        assert h2h.matches == []
        assert h2h.avg_goals_total == 0.0
        assert h2h.btts_pct == 0.0

    def test_to_dict_shape(self):
        payload = self._h2h(wide_config()).to_dict()
        assert payload["played"] == 3
        assert payload["home_wins"] == 1
        assert payload["draws"] == 1
        assert payload["away_wins"] == 1
        assert payload["avg_goals_total"] == pytest.approx(7 / 3)
        assert payload["btts_pct"] == pytest.approx(2 / 3)
        assert len(payload["matches"]) == 3


class TestGoalMarkets:

    def test_built_from_a_duck_typed_prediction(self):
        markets = GoalMarkets.from_prediction(StubPrediction(), max_scorelines=3)
        assert markets.home_xg == pytest.approx(1.8)
        assert markets.away_xg == pytest.approx(1.1)
        assert markets.total_xg == pytest.approx(2.9)
        assert markets.over_25 == pytest.approx(0.55)
        assert markets.btts_yes == pytest.approx(0.52)

    def test_scorelines_are_truncated_and_formatted(self):
        markets = GoalMarkets.from_prediction(StubPrediction(), max_scorelines=2)
        assert markets.to_dict()["top_scorelines"] == [
            {"score": "2-1", "prob": 0.11},
            {"score": "1-1", "prob": 0.1},
        ]

    def test_to_dict_shape(self):
        payload = GoalMarkets.from_prediction(StubPrediction(), max_scorelines=5).to_dict()
        assert payload["expected_goals"] == {"home": 1.8, "away": 1.1, "total": 2.9}
        assert set(payload["over_under"]) == {
            "over_1_5",
            "over_2_5",
            "over_3_5",
            "under_2_5",
        }
        assert payload["btts"] == {"yes": 0.52, "no": 0.48}
        assert len(payload["top_scorelines"]) == 5


class TestMatchInsights:

    @staticmethod
    def _fixture() -> FixtureQuery:
        return FixtureQuery(league_code="P1", home_team="Sporting", away_team="Porto")

    def test_carries_both_team_profiles(self):
        insights = calculator(wide_config()).match_insights(self._fixture())
        assert insights.home.team == "Sporting"
        assert insights.away.team == "Porto"
        assert insights.home.overall.played == 5
        assert insights.away.overall.played == 4

    def test_league_is_resolved(self):
        insights = calculator(wide_config()).match_insights(self._fixture())
        assert insights.league == "Liga Portugal"
        assert insights.league_code == "P1"

    def test_goal_markets_fall_back_to_history_without_a_market_model(self):
        markets = calculator(wide_config()).match_insights(self._fixture()).goal_markets
        assert markets is not None
        assert markets.source == GoalMarkets.SOURCE_HISTORICAL

    def test_goal_markets_fall_back_when_a_team_is_unknown_to_the_model(self):
        model = StubMarketModel(known={"Sporting"}, prediction=StubPrediction())
        markets = (
            calculator(wide_config(), market_model=model)
            .match_insights(self._fixture())
            .goal_markets
        )
        assert markets is not None
        assert markets.source == GoalMarkets.SOURCE_HISTORICAL

    def test_goal_markets_present_when_the_model_knows_both_teams(self):
        model = StubMarketModel(
            known={"Sporting", "Porto"}, prediction=StubPrediction()
        )
        insights = calculator(wide_config(), market_model=model).match_insights(
            self._fixture()
        )
        assert insights.goal_markets is not None
        assert insights.goal_markets.home_xg == pytest.approx(1.8)

    def test_goal_markets_respect_max_scorelines(self):
        model = StubMarketModel(
            known={"Sporting", "Porto"}, prediction=StubPrediction()
        )
        insights = calculator(narrow_config(), market_model=model).match_insights(
            self._fixture()
        )
        assert len(insights.goal_markets.to_dict()["top_scorelines"]) == 3

    def test_to_dict_shape(self):
        model = StubMarketModel(
            known={"Sporting", "Porto"}, prediction=StubPrediction()
        )
        payload = (
            calculator(wide_config(), market_model=model)
            .match_insights(self._fixture())
            .to_dict()
        )
        assert payload["home_team"] == "Sporting"
        assert payload["away_team"] == "Porto"
        assert payload["league"] == "Liga Portugal"
        assert payload["head_to_head"]["played"] == 3
        assert payload["home"]["overall"]["played"] == 5
        assert payload["away"]["overall"]["played"] == 4
        assert payload["goal_markets"]["expected_goals"]["home"] == pytest.approx(1.8)

    def test_to_dict_keeps_goal_markets_null_when_unavailable(self):
        """No model and no history for a side leaves nothing to approximate."""
        payload = (
            calculator(wide_config())
            .match_insights(
                FixtureQuery(league_code="P1", home_team="Sporting", away_team="Nobody")
            )
            .to_dict()
        )
        assert payload["goal_markets"] is None

    def test_to_dict_reports_the_market_source(self):
        model = StubMarketModel(known={"Sporting", "Porto"}, prediction=StubPrediction())
        payload = (
            calculator(wide_config(), market_model=model)
            .match_insights(self._fixture())
            .to_dict()
        )
        assert payload["goal_markets"]["source"] == GoalMarkets.SOURCE_MODEL


class TestKnowsTeam:

    def test_known_team_in_its_league(self):
        calc = calculator(wide_config())
        assert calc.knows_team(TeamQuery(league_code="P1", team="Sporting")) is True

    def test_unknown_team(self):
        calc = calculator(wide_config())
        assert calc.knows_team(TeamQuery(league_code="P1", team="Nobody")) is False

    def test_team_from_another_league_is_unknown(self):
        calc = calculator(wide_config())
        assert calc.knows_team(TeamQuery(league_code="P1", team="Arsenal")) is False

    def test_unknown_league_code(self):
        calc = calculator(wide_config())
        assert calc.knows_team(TeamQuery(league_code="XX", team="Sporting")) is False
