"""Tests for the historical-rates fallback behind the fixture goal markets.

The calibrated Dixon-Coles artifact is not always present in the deployed model
repo. When it is missing the markets must still be produced — from the teams'
own scoring and conceding rates — and the payload must say so, so a reader can
never mistake an approximation for the calibrated number (§7.4).

``EmpiricalPoissonModel`` deliberately implements the same ``knows``/``predict``
duck-type as the fitted model, so the calculator keeps a single code path.
"""

import pandas as pd
import pytest

from config.config_loader import InsightsConfig
from src.analysis.team_insights import (
    EmpiricalPoissonModel,
    FixtureQuery,
    GoalMarkets,
    TeamInsightsCalculator,
)

CONFIG = InsightsConfig(
    recent_matches=10, h2h_matches=10, form_sequence_length=5, max_scorelines=5
)
FIXTURE = FixtureQuery(league_code="P1", home_team="Sporting", away_team="Porto")


def _rows() -> list[tuple]:
    """A goal-rich home side and a leaky away side, so xG ordering is obvious."""
    return [
        ("2024-01-01", "Sporting", "Benfica", 3, 0),
        ("2024-01-08", "Sporting", "Guimaraes", 2, 1),
        ("2024-01-15", "Braga", "Sporting", 1, 2),
        ("2024-01-22", "Porto", "Benfica", 0, 2),
        ("2024-01-29", "Guimaraes", "Porto", 2, 1),
        ("2024-02-05", "Porto", "Braga", 1, 1),
        ("2024-02-12", "Benfica", "Braga", 1, 0),
    ]


def frame(rows: list[tuple] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(
        rows if rows is not None else _rows(),
        columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def calculator(rows: list[tuple] | None = None, **kwargs) -> TeamInsightsCalculator:
    return TeamInsightsCalculator(matches=frame(rows), config=CONFIG, **kwargs)


class StubPrediction:
    lambda_home = 9.0
    lambda_away = 9.0
    over_15 = 0.99
    over_25 = 0.98
    over_35 = 0.97
    under_25 = 0.02
    btts_yes = 0.99
    btts_no = 0.01
    top_scorelines = [(4, 4, 0.5)]


class FittedModel:
    """Stands in for the calibrated Dixon-Coles artifact."""

    def __init__(self, known: bool = True) -> None:
        self._known = known

    def knows(self, team: str) -> bool:
        return self._known

    def predict(self, home_team: str, away_team: str) -> StubPrediction:
        return StubPrediction()


class TestMarketProvenance:
    """`source` must always state where the numbers came from."""

    def test_fitted_model_is_labelled_model(self):
        insights = calculator(market_model=FittedModel()).match_insights(FIXTURE)
        assert insights.goal_markets is not None
        assert insights.goal_markets.source == GoalMarkets.SOURCE_MODEL

    def test_fallback_is_labelled_historical(self):
        insights = calculator().match_insights(FIXTURE)
        assert insights.goal_markets is not None
        assert insights.goal_markets.source == GoalMarkets.SOURCE_HISTORICAL

    def test_source_is_serialised(self):
        payload = calculator().match_insights(FIXTURE).to_dict()
        assert payload["goal_markets"]["source"] == GoalMarkets.SOURCE_HISTORICAL

    def test_fitted_model_wins_when_it_knows_both_teams(self):
        """The calibrated numbers must never be replaced by the approximation."""
        insights = calculator(market_model=FittedModel()).match_insights(FIXTURE)
        assert insights.goal_markets.home_xg == pytest.approx(9.0)

    def test_fallback_used_when_the_fitted_model_lacks_a_team(self):
        insights = calculator(market_model=FittedModel(known=False)).match_insights(
            FIXTURE
        )
        assert insights.goal_markets is not None
        assert insights.goal_markets.source == GoalMarkets.SOURCE_HISTORICAL
        assert insights.goal_markets.home_xg != pytest.approx(9.0)


class TestFallbackMarkets:

    @staticmethod
    def _markets() -> GoalMarkets:
        markets = calculator().match_insights(FIXTURE).goal_markets
        assert markets is not None
        return markets

    def test_expected_goals_are_positive(self):
        markets = self._markets()
        assert markets.home_xg > 0
        assert markets.away_xg > 0
        assert markets.total_xg == pytest.approx(markets.home_xg + markets.away_xg)

    def test_stronger_attack_earns_the_higher_expected_goals(self):
        """Sporting scored 7 in 3; Porto scored 2 in 3 and conceded 4."""
        markets = self._markets()
        assert markets.home_xg > markets.away_xg

    def test_probabilities_are_within_range(self):
        markets = self._markets()
        for value in (
            markets.over_15,
            markets.over_25,
            markets.over_35,
            markets.under_25,
            markets.btts_yes,
            markets.btts_no,
        ):
            assert 0.0 <= value <= 1.0

    def test_over_and_under_are_complementary(self):
        markets = self._markets()
        assert markets.over_25 + markets.under_25 == pytest.approx(1.0, abs=1e-6)

    def test_btts_is_complementary(self):
        markets = self._markets()
        assert markets.btts_yes + markets.btts_no == pytest.approx(1.0, abs=1e-6)

    def test_over_lines_are_monotonic(self):
        markets = self._markets()
        assert markets.over_15 >= markets.over_25 >= markets.over_35

    def test_scorelines_are_capped_by_config(self):
        config = InsightsConfig(
            recent_matches=10, h2h_matches=10, form_sequence_length=5, max_scorelines=3
        )
        calc = TeamInsightsCalculator(matches=frame(), config=config)
        markets = calc.match_insights(FIXTURE).goal_markets
        assert len(markets.to_dict()["top_scorelines"]) == 3

    def test_scorelines_are_ordered_by_probability(self):
        probs = [s["prob"] for s in self._markets().to_dict()["top_scorelines"]]
        assert probs == sorted(probs, reverse=True)


class TestFallbackAvailability:

    def test_absent_without_any_history(self):
        assert calculator([]).match_insights(FIXTURE).goal_markets is None

    def test_absent_when_a_team_has_no_history(self):
        insights = calculator().match_insights(
            FixtureQuery(league_code="P1", home_team="Sporting", away_team="Nobody")
        )
        assert insights.goal_markets is None

    def test_present_for_a_team_with_only_away_history(self):
        rows = [
            ("2024-01-01", "Braga", "Sporting", 1, 2),
            ("2024-01-08", "Porto", "Benfica", 0, 2),
        ]
        insights = calculator(rows).match_insights(FIXTURE)
        assert insights.goal_markets is not None


class TestEmpiricalPoissonModelInterface:
    """It must be substitutable for the fitted model (§4.5)."""

    @staticmethod
    def _model() -> EmpiricalPoissonModel:
        return EmpiricalPoissonModel(
            matches=frame(), league_average_goals=1.35, max_goals=6
        )

    def test_knows_a_team_present_in_the_frame(self):
        assert self._model().knows("Sporting") is True

    def test_does_not_know_an_absent_team(self):
        assert self._model().knows("Nobody") is False

    def test_predict_returns_a_dixon_coles_shaped_object(self):
        prediction = self._model().predict("Sporting", "Porto")
        for attribute in (
            "lambda_home",
            "lambda_away",
            "over_15",
            "over_25",
            "over_35",
            "under_25",
            "btts_yes",
            "btts_no",
            "top_scorelines",
        ):
            assert hasattr(prediction, attribute), attribute

    def test_prediction_feeds_goal_markets_unchanged(self):
        prediction = self._model().predict("Sporting", "Porto")
        markets = GoalMarkets.from_prediction(prediction, max_scorelines=3)
        assert markets.total_xg > 0

    def test_expected_goals_are_clamped(self):
        """A freak scoreline must not produce an absurd expectation."""
        rows = [
            ("2024-01-01", "Sporting", "Porto", 15, 0),
            ("2024-01-08", "Sporting", "Porto", 14, 0),
        ]
        model = EmpiricalPoissonModel(
            matches=frame(rows), league_average_goals=1.35, max_goals=6
        )
        prediction = model.predict("Sporting", "Porto")
        assert prediction.lambda_home <= EmpiricalPoissonModel.MAX_EXPECTED_GOALS
        assert prediction.lambda_away >= EmpiricalPoissonModel.MIN_EXPECTED_GOALS
