"""Edge cases and error paths for the team / head-to-head insight calculator.

The calculator sits behind a live endpoint, so every degenerate input a real
data set can produce — an unplayed fixture, a season without shot data, a team
that has only ever played away — must yield a usable answer instead of an
exception.
"""

import pandas as pd
import pytest

from config.config_loader import InsightsConfig
from src.analysis.team_insights import (
    FixtureQuery,
    GoalMarkets,
    TeamInsightsCalculator,
    TeamQuery,
)

CONFIG = InsightsConfig(
    recent_matches=5, h2h_matches=5, form_sequence_length=5, max_scorelines=3
)
SPORTING = TeamQuery(league_code="P1", team="Sporting")


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def minimal_rows() -> list[dict]:
    """Only the columns every football-data.co.uk season is guaranteed to have."""
    return [
        {
            "Date": "2024-01-01",
            "HomeTeam": "Sporting",
            "AwayTeam": "Porto",
            "FTHG": 2,
            "FTAG": 1,
        },
        {
            "Date": "2024-01-08",
            "HomeTeam": "Porto",
            "AwayTeam": "Sporting",
            "FTHG": 0,
            "FTAG": 3,
        },
    ]


def calculator(rows: list[dict], **kwargs) -> TeamInsightsCalculator:
    return TeamInsightsCalculator(matches=frame(rows), config=CONFIG, **kwargs)


class TestEmptyData:

    def test_empty_frame_yields_a_zeroed_profile(self):
        insights = calculator([]).team_insights(SPORTING)
        assert insights.overall.played == 0
        assert insights.recent_matches == []
        assert insights.form_sequence == []
        assert insights.rates.btts == 0.0
        assert insights.averages.shots == 0.0

    def test_empty_frame_yields_an_empty_head_to_head(self):
        insights = calculator([]).match_insights(
            FixtureQuery(league_code="P1", home_team="Sporting", away_team="Porto")
        )
        assert insights.head_to_head.played == 0
        assert insights.goal_markets is None

    def test_empty_frame_knows_no_team(self):
        assert calculator([]).knows_team(SPORTING) is False

    def test_empty_frame_still_serialises(self):
        payload = calculator([]).team_insights(SPORTING).to_dict()
        assert payload["overall"]["played"] == 0
        assert payload["recent_matches"] == []

    def test_unknown_team_yields_a_zeroed_profile(self):
        insights = calculator(minimal_rows()).team_insights(
            TeamQuery(league_code="P1", team="Nobody")
        )
        assert insights.overall.played == 0
        assert insights.team == "Nobody"


class TestOptionalColumns:
    """Seasons without shot/corner/card columns must not break averages."""

    def test_missing_stat_columns_zero_the_averages(self):
        averages = calculator(minimal_rows()).team_insights(SPORTING).averages
        assert averages.shots == 0.0
        assert averages.shots_on_target == 0.0
        assert averages.corners == 0.0
        assert averages.cards == 0.0

    def test_records_still_computed_without_stat_columns(self):
        overall = calculator(minimal_rows()).team_insights(SPORTING).overall
        assert (overall.played, overall.wins) == (2, 2)

    def test_half_present_column_pair_is_ignored(self):
        """A home column without its away twin cannot be attributed safely."""
        rows = [dict(row, HS=10) for row in minimal_rows()]
        assert calculator(rows).team_insights(SPORTING).averages.shots == 0.0

    def test_missing_date_column_yields_empty_dates(self):
        rows = [
            {k: v for k, v in row.items() if k != "Date"} for row in minimal_rows()
        ]
        records = calculator(rows).team_insights(SPORTING).recent_matches
        assert [record.date for record in records] == ["", ""]

    def test_unparseable_date_yields_an_empty_date(self):
        """A row whose date failed to parse still counts, just undated."""
        rows = minimal_rows()
        df = frame(rows)
        df.loc[0, "Date"] = pd.NaT
        calc = TeamInsightsCalculator(matches=df, config=CONFIG)
        records = calc.team_insights(SPORTING).recent_matches
        assert "" in [record.date for record in records]
        assert calc.team_insights(SPORTING).overall.played == 2

    def test_nan_stat_value_is_excluded_from_the_average(self):
        rows = minimal_rows()
        rows[0].update({"HS": 10, "AS": 8})
        rows[1].update({"HS": None, "AS": None})
        assert calculator(rows).team_insights(SPORTING).averages.shots == pytest.approx(
            10.0
        )


class TestIncompleteRows:

    def test_unplayed_fixtures_are_ignored(self):
        rows = minimal_rows() + [
            {
                "Date": "2024-02-01",
                "HomeTeam": "Sporting",
                "AwayTeam": "Benfica",
                "FTHG": None,
                "FTAG": None,
            }
        ]
        insights = calculator(rows).team_insights(SPORTING)
        assert insights.overall.played == 2
        assert len(insights.recent_matches) == 2

    def test_unplayed_meetings_are_ignored_in_head_to_head(self):
        rows = minimal_rows() + [
            {
                "Date": "2024-02-01",
                "HomeTeam": "Sporting",
                "AwayTeam": "Porto",
                "FTHG": None,
                "FTAG": None,
            }
        ]
        h2h = calculator(rows).match_insights(
            FixtureQuery(league_code="P1", home_team="Sporting", away_team="Porto")
        ).head_to_head
        assert h2h.played == 2

    def test_unsorted_input_is_ordered_by_date(self):
        rows = list(reversed(minimal_rows()))
        records = calculator(rows).team_insights(SPORTING).recent_matches
        assert [record.date for record in records] == ["2024-01-08", "2024-01-01"]


class TestVenueOnlyHistories:

    def test_team_with_only_home_matches(self):
        rows = [minimal_rows()[0]]
        insights = calculator(rows).team_insights(SPORTING)
        assert insights.home.played == 1
        assert insights.away.played == 0
        assert insights.away.points_per_game == 0.0

    def test_team_with_only_away_matches(self):
        rows = [minimal_rows()[1]]
        insights = calculator(rows).team_insights(SPORTING)
        assert insights.away.played == 1
        assert insights.home.played == 0


class TestConfigurationVariations:

    @staticmethod
    def _with(config: InsightsConfig) -> TeamInsightsCalculator:
        return TeamInsightsCalculator(matches=frame(minimal_rows()), config=config)

    def test_zero_window_produces_an_empty_recent_block(self):
        config = InsightsConfig(
            recent_matches=0, h2h_matches=0, form_sequence_length=0, max_scorelines=0
        )
        insights = self._with(config).team_insights(SPORTING)
        assert insights.recent.played == 0
        assert insights.recent_matches == []
        assert insights.form_sequence == []
        # The all-time record ignores the window entirely.
        assert insights.overall.played == 2

    def test_form_sequence_never_exceeds_the_available_history(self):
        config = InsightsConfig(
            recent_matches=50, h2h_matches=50, form_sequence_length=50, max_scorelines=5
        )
        assert self._with(config).team_insights(SPORTING).form_sequence == ["W", "W"]

    def test_h2h_cap_larger_than_history_returns_everything(self):
        config = InsightsConfig(
            recent_matches=5, h2h_matches=99, form_sequence_length=5, max_scorelines=5
        )
        h2h = self._with(config).match_insights(
            FixtureQuery(league_code="P1", home_team="Sporting", away_team="Porto")
        ).head_to_head
        assert len(h2h.matches) == 2


class TestLeagueResolution:

    def test_unlabelled_frame_serves_any_league_code(self):
        """A single-competition frame has no League column to filter on."""
        insights = calculator(minimal_rows()).team_insights(
            TeamQuery(league_code="ANY", team="Sporting")
        )
        assert insights.overall.played == 2
        assert insights.league == "ANY"

    def test_unknown_code_falls_back_to_the_code_as_label(self):
        insights = calculator(minimal_rows(), leagues={"P1": "Liga Portugal"}).team_insights(
            TeamQuery(league_code="ZZ", team="Sporting")
        )
        assert insights.league == "ZZ"

    def test_labelled_frame_with_unknown_code_yields_no_matches(self):
        rows = [dict(row, League="Liga Portugal") for row in minimal_rows()]
        insights = calculator(rows, leagues={"P1": "Liga Portugal"}).team_insights(
            TeamQuery(league_code="ZZ", team="Sporting")
        )
        assert insights.overall.played == 0


class StubPrediction:
    lambda_home = 1.5
    lambda_away = 1.0
    over_15 = 0.7
    over_25 = 0.5
    over_35 = 0.3
    under_25 = 0.5
    btts_yes = 0.5
    btts_no = 0.5
    top_scorelines: list[tuple[int, int, float]] = []


class RecordingModel:
    def __init__(self, known: bool, prediction: object) -> None:
        self._known = known
        self._prediction = prediction
        self.calls: list[tuple[str, str]] = []

    def knows(self, team: str) -> bool:
        return self._known

    def predict(self, home_team: str, away_team: str) -> object:
        self.calls.append((home_team, away_team))
        return self._prediction


class TestGoalMarketEdges:

    @staticmethod
    def _fixture() -> FixtureQuery:
        return FixtureQuery(league_code="P1", home_team="Sporting", away_team="Porto")

    def test_model_that_knows_neither_team_is_not_called(self):
        model = RecordingModel(known=False, prediction=StubPrediction())
        insights = calculator(minimal_rows(), market_model=model).match_insights(
            self._fixture()
        )
        assert model.calls == []
        assert insights.goal_markets.source == GoalMarkets.SOURCE_HISTORICAL

    def test_model_returning_nothing_degrades_to_historical_markets(self):
        model = RecordingModel(known=True, prediction=None)
        insights = calculator(minimal_rows(), market_model=model).match_insights(
            self._fixture()
        )
        assert insights.goal_markets is not None
        assert insights.goal_markets.source == GoalMarkets.SOURCE_HISTORICAL

    def test_empty_scoreline_list_is_serialised_as_empty(self):
        markets = GoalMarkets.from_prediction(StubPrediction(), max_scorelines=5)
        assert markets.to_dict()["top_scorelines"] == []

    def test_zero_max_scorelines_drops_every_scoreline(self):
        prediction = StubPrediction()
        prediction.top_scorelines = [(1, 0, 0.2)]
        markets = GoalMarkets.from_prediction(prediction, max_scorelines=0)
        assert markets.to_dict()["top_scorelines"] == []

    def test_markets_are_independent_of_the_historical_frame(self):
        """A team with no history still gets markets when the model knows it."""
        model = RecordingModel(known=True, prediction=StubPrediction())
        insights = calculator([], market_model=model).match_insights(self._fixture())
        assert insights.goal_markets is not None
        assert insights.goal_markets.total_xg == pytest.approx(2.5)


class TestSameTeamFixture:
    """A degenerate query the UI prevents but the API must survive."""

    def test_team_against_itself_does_not_crash(self):
        insights = calculator(minimal_rows()).match_insights(
            FixtureQuery(league_code="P1", home_team="Sporting", away_team="Sporting")
        )
        assert insights.head_to_head.played == 2
        assert insights.home.team == insights.away.team
