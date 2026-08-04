"""Tests for N/A odds handling in the inference response payload."""

import importlib

from src.models.predictor import MatchPrediction
from src.scrapers.fixtures_fetcher import Fixture

run_inference = importlib.import_module("scripts.run_inference")


def _fixture(with_odds: bool) -> Fixture:
    return Fixture(
        division="E0",
        league="Premier League",
        date="01/01/2024",
        time="15:00",
        home_team="Arsenal",
        away_team="Chelsea",
        b365_home=2.0 if with_odds else None,
        b365_draw=3.5 if with_odds else None,
        b365_away=4.0 if with_odds else None,
    )


def _prediction() -> MatchPrediction:
    return MatchPrediction(
        home_team="Arsenal",
        away_team="Chelsea",
        home_win_prob=0.5,
        draw_prob=0.25,
        away_win_prob=0.25,
        predicted_outcome="Home Win",
        confidence=0.5,
    )


class TestPayloadOddsHandling:
    def test_odds_null_when_no_odds(self) -> None:
        payload = run_inference._build_response_payload(
            "E0",
            "Premier League",
            [_fixture(with_odds=False)],
            [_prediction()],
            [None],
            [],
        )
        match = payload["matches"][0]
        assert match["odds"] is None
        assert match["implied_probabilities"] is None
        assert match["value_bets"] == []

    def test_odds_present_when_odds(self) -> None:
        payload = run_inference._build_response_payload(
            "E0",
            "Premier League",
            [_fixture(with_odds=True)],
            [_prediction()],
            [None],
            [],
        )
        match = payload["matches"][0]
        assert match["odds"] == {"home": 2.0, "draw": 3.5, "away": 4.0}
        assert match["implied_probabilities"] is not None
        # Payload rounds implied probabilities to 4 decimals.
        assert abs(sum(match["implied_probabilities"].values()) - 1.0) < 1e-3
