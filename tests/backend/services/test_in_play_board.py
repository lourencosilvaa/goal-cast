"""Joining the live board to the stored predictions.

The calculator in :mod:`src.backend.services.in_play` re-prices one match given
its pre-match scoring rates and its current state. Getting those two things
into the same object is the job under test here, and it is not arithmetic — it
is an identity problem.

The two sides are spelled differently. Predictions are keyed by
football-data.co.uk names ("Sp Lisbon"), the live board reports whatever the
provider calls a club ("Sporting CP"). Guessing between them is exactly what
:mod:`src.teams.resolver` exists to refuse, so a name that does not resolve
must leave the match unpriced and *say so* — a silently shorter list is a
feature that looks broken only to the person who knows how many matches are
being played.
"""

import pytest

from src.backend.services.in_play_board import (
    InPlayBoard,
    NameMatcher,
    PredictionSource,
)
from src.contracts.results import LiveResultsResponse, MatchModel

_FETCHED = "2026-08-09T17:30:00"


def _match(
    league: str = "P1",
    home: str = "Porto",
    away: str = "Alverca",
    status: str = "live",
    home_goals: int | None = 1,
    away_goals: int | None = 0,
    elapsed: int = 60,
    estimated: bool = False,
) -> MatchModel:
    return MatchModel(
        league=league,
        kickoff="2026-08-09T17:00:00",
        home_team=home,
        away_team=away,
        status=status,
        home_goals=home_goals,
        away_goals=away_goals,
        elapsed_minutes=elapsed,
        minute_estimated=estimated,
    )


def _prediction(
    home: str = "Porto",
    away: str = "Alverca",
    xg: tuple[float, float] | None = (2.1, 0.8),
    probabilities: dict | None = None,
) -> dict:
    row: dict = {
        "home_team": home,
        "away_team": away,
        "probabilities": probabilities
        or {"home_win": 0.72, "draw": 0.18, "away_win": 0.10},
        "time": "18:00",
    }
    if xg is not None:
        row["expected_goals"] = {"home": xg[0], "away": xg[1], "total": sum(xg)}
    return row


class _Gateway:
    """A live board, or a failure."""

    def __init__(self, matches=None, stale: bool = False, error: Exception | None = None):
        self._matches = matches if matches is not None else [_match()]
        self._stale = stale
        self._error = error
        self.calls: list[list[str]] = []

    def live(self, leagues):
        self.calls.append(list(leagues))
        if self._error:
            raise self._error
        return LiveResultsResponse(
            fetched_at=_FETCHED,
            source="football-data.org",
            stale=self._stale,
            matches=self._matches,
            events=[],
        )

    def history(self, league, season):  # pragma: no cover - unused here
        raise AssertionError("the board never asks for history")


class _Predictions(PredictionSource):
    def __init__(self, rows=None):
        self._rows = rows if rows is not None else {"P1": [_prediction()]}
        self.asked: list[tuple[str, str]] = []

    def matches(self, league: str, match_date: str):
        self.asked.append((league, match_date))
        return self._rows.get(league, [])


class _Names(NameMatcher):
    """Resolves by an explicit table; anything absent is unresolved.

    Stated per test rather than inherited from the real registry, so a test
    says which names it considers known instead of depending on what the
    shipped alias seed happens to contain.
    """

    def __init__(self, table: dict[str, str] | None = None):
        self._table = table or {}

    def canonical(self, league: str, raw_name: str) -> str | None:
        return self._table.get(raw_name, raw_name if not self._table else None)


def _board(gateway=None, predictions=None, names=None) -> InPlayBoard:
    return InPlayBoard(
        gateway=gateway or _Gateway(),
        predictions=predictions or _Predictions(),
        names=names or _Names(),
    )


class TestTheHappyPath:
    def test_a_live_match_with_a_stored_prediction_is_priced(self):
        assert len(_board().build([]).matches) == 1

    def test_the_priced_match_uses_the_canonical_names(self):
        """The frontend joins on the prediction's spelling, not the board's."""
        gateway = _Gateway([_match(home="FC Porto")])
        names = _Names({"FC Porto": "Porto", "Alverca": "Alverca"})
        priced = _board(gateway=gateway, names=names).build([]).matches[0]
        assert (priced.home_team, priced.away_team) == ("Porto", "Alverca")

    def test_the_live_score_reaches_the_forecast(self):
        """1-0 up at 60 minutes must beat the pre-match 72%."""
        priced = _board().build([]).matches[0]
        assert priced.forecast.outcome.home_win > 0.72

    def test_the_pre_match_probabilities_are_carried_alongside(self):
        """Both numbers are shown together — the movement is the point."""
        assert _board().build([]).matches[0].pre_match.home_win == pytest.approx(0.72)

    def test_the_scoreboard_is_carried(self):
        priced = _board().build([]).matches[0]
        assert (priced.home_goals, priced.away_goals) == (1, 0)

    def test_the_estimated_minute_flag_survives(self):
        """A UI must be able to write ``~60'`` rather than pass arithmetic off
        as the official clock."""
        gateway = _Gateway([_match(estimated=True)])
        assert _board(gateway=gateway).build([]).matches[0].minute_estimated is True

    def test_the_board_reports_when_it_was_fetched(self):
        assert _board().build([]).fetched_at is not None

    def test_staleness_is_carried_through(self):
        assert _board(gateway=_Gateway(stale=True)).build([]).stale is True


class TestWhichMatchesQualify:
    def test_a_scheduled_match_is_not_priced(self):
        """Before kick-off the stored prediction *is* the answer."""
        gateway = _Gateway([_match(status="scheduled", home_goals=None, away_goals=None)])
        assert _board(gateway=gateway).build([]).matches == []

    def test_a_finished_match_is_not_priced(self):
        gateway = _Gateway([_match(status="finished")])
        assert _board(gateway=gateway).build([]).matches == []

    def test_a_paused_match_is_priced(self):
        """Half-time is still a match in progress with a score."""
        gateway = _Gateway([_match(status="paused", elapsed=45)])
        assert len(_board(gateway=gateway).build([]).matches) == 1

    def test_a_live_match_without_a_score_is_not_priced(self):
        """``None`` is not 0-0; pricing it as one would invent a scoreline."""
        gateway = _Gateway([_match(home_goals=None, away_goals=None)])
        assert _board(gateway=gateway).build([]).matches == []


class TestWhatCannotBePriced:
    def test_an_unresolved_name_leaves_the_match_unpriced(self):
        names = _Names({"Alverca": "Alverca"})
        board = _board(gateway=_Gateway([_match(home="Unknown FC")]), names=names)
        assert board.build([]).matches == []

    def test_an_unresolved_name_is_reported_not_dropped(self):
        names = _Names({"Alverca": "Alverca"})
        board = _board(gateway=_Gateway([_match(home="Unknown FC")]), names=names)
        unpriced = board.build([]).unpriced
        assert len(unpriced) == 1 and unpriced[0].reason == InPlayBoard.UNKNOWN_TEAM

    def test_the_report_names_the_match_as_the_provider_spelled_it(self):
        """Whoever debugs this needs the spelling that failed to resolve."""
        names = _Names({"Alverca": "Alverca"})
        board = _board(gateway=_Gateway([_match(home="Unknown FC")]), names=names)
        assert board.build([]).unpriced[0].home_team == "Unknown FC"

    def test_a_match_with_no_stored_prediction_is_reported(self):
        gateway = _Gateway([_match(home="Porto", away="Braga")])
        board = _board(gateway=gateway, predictions=_Predictions({"P1": []}))
        assert board.build([]).unpriced[0].reason == InPlayBoard.NO_PREDICTION

    def test_a_prediction_without_expected_goals_cannot_be_priced(self):
        """The Poisson rates are the model's input; there is no substitute."""
        predictions = _Predictions({"P1": [_prediction(xg=None)]})
        board = _board(predictions=predictions)
        assert board.build([]).unpriced[0].reason == InPlayBoard.NO_EXPECTED_GOALS

    def test_an_unpriceable_match_does_not_stop_the_others(self):
        gateway = _Gateway([_match(home="Unknown FC"), _match()])
        names = _Names({"Porto": "Porto", "Alverca": "Alverca"})
        board = _board(gateway=gateway, names=names)
        result = board.build([])
        assert len(result.matches) == 1 and len(result.unpriced) == 1


class TestTheQueriesItMakes:
    def test_the_requested_leagues_are_forwarded_to_the_gateway(self):
        gateway = _Gateway()
        _board(gateway=gateway).build(["P1", "E0"])
        assert gateway.calls == [["P1", "E0"]]

    def test_predictions_are_only_read_for_leagues_actually_playing(self):
        """One Supabase read per live league, not per served league."""
        predictions = _Predictions()
        _board(predictions=predictions).build(["P1", "E0", "SP1"])
        assert predictions.asked == [("P1", "09/08/2026")]

    def test_the_day_asked_for_is_the_day_the_match_kicked_off(self):
        """Not "today": the server's clock can already be past midnight while
        a match that kicked off at 21:00 is still being played, and the
        prediction is filed under the day of the fixture."""
        gateway = _Gateway(
            [
                MatchModel(
                    league="P1",
                    kickoff="2026-08-09T20:45:00",
                    home_team="Porto",
                    away_team="Alverca",
                    status="live",
                    home_goals=0,
                    away_goals=0,
                    elapsed_minutes=80,
                )
            ]
        )
        predictions = _Predictions()
        _board(gateway=gateway, predictions=predictions).build([])
        assert predictions.asked == [("P1", "09/08/2026")]

    def test_two_matches_in_one_league_share_a_single_read(self):
        gateway = _Gateway([_match(), _match(home="Braga", away="Estoril")])
        predictions = _Predictions()
        _board(gateway=gateway, predictions=predictions).build([])
        assert predictions.asked == [("P1", "09/08/2026")]


class TestMalformedRows:
    """The rows are JSON written by a separate offline process. A shape this
    reader did not expect must cost one match, never the request."""

    def test_expected_goals_that_are_not_numbers_leave_the_match_unpriced(self):
        row = _prediction()
        row["expected_goals"] = {"home": "2.1", "away": "0.8", "total": "2.9"}
        board = _board(predictions=_Predictions({"P1": [row]}))
        assert board.build([]).unpriced[0].reason == InPlayBoard.NO_EXPECTED_GOALS

    def test_expected_goals_that_are_not_a_mapping_are_rejected(self):
        row = _prediction()
        row["expected_goals"] = 2.9
        board = _board(predictions=_Predictions({"P1": [row]}))
        assert board.build([]).unpriced[0].reason == InPlayBoard.NO_EXPECTED_GOALS

    def test_a_half_filled_expected_goals_is_rejected_not_halved(self):
        row = _prediction()
        row["expected_goals"] = {"home": 2.1}
        board = _board(predictions=_Predictions({"P1": [row]}))
        assert board.build([]).unpriced[0].reason == InPlayBoard.NO_EXPECTED_GOALS

    def test_a_row_without_probabilities_is_still_priced(self):
        """The live number needs only the scoring rates. A missing pre-match
        triple costs the comparison, not the forecast."""
        row = _prediction()
        del row["probabilities"]
        priced = _board(predictions=_Predictions({"P1": [row]})).build([]).matches[0]
        assert priced.forecast.outcome.home_win > 0.5
        assert priced.pre_match.home_win == 0.0

    def test_probabilities_of_the_wrong_shape_do_not_raise(self):
        row = _prediction()
        row["probabilities"] = "0.72/0.18/0.10"
        priced = _board(predictions=_Predictions({"P1": [row]})).build([]).matches[0]
        assert priced.pre_match.draw == 0.0

    def test_a_row_missing_its_team_names_simply_never_matches(self):
        board = _board(predictions=_Predictions({"P1": [{"expected_goals": {}}]}))
        assert board.build([]).unpriced[0].reason == InPlayBoard.NO_PREDICTION


class TestFixtureIdentity:
    def test_the_reverse_fixture_is_not_the_same_match(self):
        """Two teams meet twice a season; only the ordered pair identifies one."""
        predictions = _Predictions({"P1": [_prediction(home="Alverca", away="Porto")]})
        board = _board(predictions=predictions)
        assert board.build([]).unpriced[0].reason == InPlayBoard.NO_PREDICTION

    def test_the_right_row_is_chosen_from_a_full_matchday(self):
        rows = [
            _prediction(home="Braga", away="Estoril", xg=(3.0, 0.2)),
            _prediction(home="Porto", away="Alverca", xg=(2.1, 0.8)),
        ]
        priced = _board(predictions=_Predictions({"P1": rows})).build([]).matches[0]
        assert priced.forecast.expected_away_goals == pytest.approx(0.8 * (30 / 90))


class TestFailuresPropagate:
    def test_a_gateway_failure_is_not_swallowed(self):
        """An empty board and an unreachable service must not look alike."""
        gateway = _Gateway(error=RuntimeError("results service unreachable"))
        with pytest.raises(RuntimeError):
            _board(gateway=gateway).build([])
