"""Re-pricing a match that is already under way.

The pre-match model answers "who wins?" before a ball is kicked. Once a match
is running, two things it never saw are known: the score, and how little time
is left to change it. Conditioning on both is what makes a live number
different from the stored one — and the pre-match prediction, replayed at
minute 80 of a 2-0, is simply wrong.

The model is the standard one: goals arrive as a Poisson process at the rate
the pre-match model expected, so the goals *still to come* are Poisson with
that rate scaled by the fraction of the match remaining. The current score is
a known head start.

Its limits are worth stating because they are visible in the output. It
assumes the scoring rate is unaffected by the score itself, which is false —
a leading side defends — so comebacks are slightly under-predicted. It knows
nothing of red cards or injuries. It is a re-pricing, not a simulation.
"""

import pytest

from src.backend.services.in_play import (
    ExpectedGoals,
    InPlayCalculator,
    LiveState,
)

_EVEN = ExpectedGoals(home=1.4, away=1.2)


def _probs(expected=_EVEN, home=0, away=0, minute=0):
    return InPlayCalculator().outcome(
        expected, LiveState(home_goals=home, away_goals=away, elapsed_minutes=minute)
    )


class TestShape:
    def test_the_three_outcomes_sum_to_one(self):
        p = _probs(minute=30, home=1, away=0)
        assert p.home_win + p.draw + p.away_win == pytest.approx(1.0)

    def test_every_probability_is_a_probability(self):
        p = _probs(minute=70, home=2, away=1)
        assert all(0.0 <= value <= 1.0 for value in (p.home_win, p.draw, p.away_win))


class TestBeforeKickoff:
    def test_at_minute_zero_the_favourite_is_the_stronger_side(self):
        p = _probs(expected=ExpectedGoals(home=2.2, away=0.7))
        assert p.home_win > p.away_win

    def test_a_goalless_start_leaves_a_real_draw_chance(self):
        assert _probs().draw > 0.15

    def test_symmetric_rates_give_symmetric_odds(self):
        p = _probs(expected=ExpectedGoals(home=1.3, away=1.3))
        assert p.home_win == pytest.approx(p.away_win)


class TestTheScoreMatters:
    def test_leading_raises_the_chance_of_winning(self):
        goalless = _probs(minute=30)
        ahead = _probs(minute=30, home=1, away=0)
        assert ahead.home_win > goalless.home_win

    def test_a_two_goal_lead_beats_a_one_goal_lead(self):
        one = _probs(minute=60, home=1, away=0)
        two = _probs(minute=60, home=2, away=0)
        assert two.home_win > one.home_win

    def test_trailing_lowers_the_chance_of_winning(self):
        assert _probs(minute=60, home=0, away=1).home_win < _probs(minute=60).home_win

    def test_a_level_score_late_makes_the_draw_likeliest(self):
        p = _probs(minute=85, home=1, away=1)
        assert p.draw > p.home_win and p.draw > p.away_win


class TestTimeMatters:
    def test_the_same_lead_is_safer_later(self):
        early = _probs(minute=15, home=1, away=0)
        late = _probs(minute=80, home=1, away=0)
        assert late.home_win > early.home_win

    def test_a_deficit_is_harder_to_overturn_later(self):
        early = _probs(minute=15, home=0, away=1)
        late = _probs(minute=80, home=0, away=1)
        assert late.home_win < early.home_win

    def test_at_full_time_the_result_is_certain(self):
        p = _probs(minute=90, home=2, away=0)
        assert p.home_win == pytest.approx(1.0)

    def test_a_draw_at_full_time_is_a_certain_draw(self):
        assert _probs(minute=90, home=1, away=1).draw == pytest.approx(1.0)

    def test_the_final_whistle_cannot_be_overshot(self):
        """Elapsed minutes are estimated upstream and can arrive above 90."""
        assert _probs(minute=120, home=0, away=1).away_win == pytest.approx(1.0)

    def test_a_negative_minute_is_treated_as_kick_off(self):
        assert _probs(minute=-5).home_win == pytest.approx(_probs(minute=0).home_win)


class TestDegenerateRates:
    def test_a_zero_rate_side_can_still_hold_a_lead(self):
        p = _probs(expected=ExpectedGoals(home=0.0, away=0.0), minute=45, home=1, away=0)
        assert p.home_win == pytest.approx(1.0)

    def test_a_zero_rate_goalless_match_stays_goalless(self):
        p = _probs(expected=ExpectedGoals(home=0.0, away=0.0), minute=45)
        assert p.draw == pytest.approx(1.0)

    def test_a_negative_rate_is_treated_as_zero(self):
        """Stored expected goals come from a payload; a bad one must not make
        the arithmetic explode."""
        p = _probs(expected=ExpectedGoals(home=-1.0, away=-1.0), minute=45, home=1)
        assert p.home_win == pytest.approx(1.0)


class TestGoalMarkets:
    def test_the_expected_final_score_includes_goals_already_scored(self):
        forecast = InPlayCalculator().forecast(
            _EVEN, LiveState(home_goals=2, away_goals=0, elapsed_minutes=45)
        )
        assert forecast.expected_home_goals > 2.0

    def test_the_remaining_share_is_reported(self):
        forecast = InPlayCalculator().forecast(
            _EVEN, LiveState(home_goals=0, away_goals=0, elapsed_minutes=45)
        )
        assert forecast.remaining_fraction == pytest.approx(0.5)

    def test_no_time_left_means_no_more_goals_expected(self):
        forecast = InPlayCalculator().forecast(
            _EVEN, LiveState(home_goals=1, away_goals=1, elapsed_minutes=90)
        )
        assert forecast.expected_home_goals == pytest.approx(1.0)
