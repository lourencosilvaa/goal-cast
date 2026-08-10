"""Orchestration: validate the request, ask the right chain, attribute the answer.

The validation is the interesting part. A league or season nobody configured
must produce an explicit refusal, never an empty list — an empty list is
indistinguishable from "that league played no matches", and it is the answer
that would be returned forever if a code were misspelled.
"""

from datetime import datetime

import pytest

from config.config_loader import ResultsLiveConfig
from src.results_service.service import (
    ResultsCatalogue,
    ResultsService,
    UnknownLeagueError,
    UnknownSeasonError,
)
from src.scrapers.results.live_tracker import LiveResultsTracker
from src.scrapers.results.models import HistoryQuery, MatchResult, MatchStatus

_CATALOGUE = ResultsCatalogue(leagues=("P1", "E0", "CL"), seasons=("2526", "2425"))


def _match(league="P1", home="Porto") -> MatchResult:
    return MatchResult(
        league=league,
        kickoff=datetime(2026, 8, 9, 17, 0),
        home_team=home,
        away_team="Alverca",
        status=MatchStatus.FINISHED,
        home_goals=1,
        away_goals=0,
        source="stub",
    )


class _HistoryChain:
    def __init__(self, matches=None, source="local-corpus"):
        self._matches = matches if matches is not None else [_match()]
        self.last_source = source
        self.queries: list[HistoryQuery] = []

    def fetch_history(self, query):
        self.queries.append(query)
        return list(self._matches)


class _LiveChain:
    def __init__(self, matches=None):
        self._matches = matches if matches is not None else [_match()]
        self.last_source = "football-data.org"
        self.requested: list[list[str]] = []

    def fetch_live(self, leagues):
        self.requested.append(list(leagues))
        return list(self._matches)


def _service(history=None, live=None, catalogue=_CATALOGUE) -> ResultsService:
    tracker = LiveResultsTracker(
        live or _LiveChain(),
        ResultsLiveConfig(poll_interval_seconds=60, stale_after_seconds=300),
        clock=lambda: datetime(2026, 8, 9, 17, 30),
    )
    return ResultsService(
        history=history or _HistoryChain(), tracker=tracker, catalogue=catalogue
    )


class TestHistory:
    def test_a_configured_league_and_season_are_served(self):
        result = _service().history(HistoryQuery(league="P1", season="2526"))
        assert len(result.matches) == 1

    def test_the_answering_source_is_reported(self):
        result = _service().history(HistoryQuery(league="P1", season="2526"))
        assert result.source == "local-corpus"

    def test_the_query_is_echoed_back(self):
        result = _service().history(HistoryQuery(league="P1", season="2526"))
        assert (result.league, result.season) == ("P1", "2526")

    def test_an_unknown_league_is_refused_explicitly(self):
        with pytest.raises(UnknownLeagueError, match="ZZ"):
            _service().history(HistoryQuery(league="ZZ", season="2526"))

    def test_an_unknown_season_is_refused_explicitly(self):
        with pytest.raises(UnknownSeasonError, match="1999"):
            _service().history(HistoryQuery(league="P1", season="1999"))

    def test_a_refusal_names_what_would_have_been_valid(self):
        """A bare "unknown league" leaves the caller guessing at spelling."""
        with pytest.raises(UnknownLeagueError, match="P1"):
            _service().history(HistoryQuery(league="ZZ", season="2526"))

    def test_an_unknown_league_never_reaches_the_chain(self):
        chain = _HistoryChain()
        with pytest.raises(UnknownLeagueError):
            _service(history=chain).history(HistoryQuery(league="ZZ", season="2526"))
        assert chain.queries == []

    def test_a_configured_league_with_no_matches_is_an_empty_answer(self):
        """Empty is a legitimate answer once the query itself is valid."""
        service = _service(history=_HistoryChain(matches=[], source=None))
        result = service.history(HistoryQuery(league="P1", season="2425"))
        assert result.matches == ()

    def test_an_unanswered_query_reports_no_source(self):
        service = _service(history=_HistoryChain(matches=[], source=None))
        result = service.history(HistoryQuery(league="P1", season="2425"))
        assert result.source == ""


class TestLive:
    def test_the_requested_leagues_are_passed_to_the_chain(self):
        chain = _LiveChain()
        _service(live=chain).live(["P1", "E0"])
        assert chain.requested == [["P1", "E0"]]

    def test_an_unknown_league_is_refused_explicitly(self):
        with pytest.raises(UnknownLeagueError, match="ZZ"):
            _service().live(["P1", "ZZ"])

    def test_an_unknown_league_never_reaches_the_chain(self):
        chain = _LiveChain()
        with pytest.raises(UnknownLeagueError):
            _service(live=chain).live(["ZZ"])
        assert chain.requested == []

    def test_no_leagues_means_every_configured_one(self):
        """The score board's default view is "everything on today"."""
        chain = _LiveChain()
        _service(live=chain).live([])
        assert chain.requested == [["P1", "E0", "CL"]]

    def test_the_update_carries_the_snapshot_and_its_staleness(self):
        update = _service().live(["P1"])
        assert update.stale is False
        assert len(update.snapshot.matches) == 1

    def test_duplicate_leagues_are_requested_once(self):
        chain = _LiveChain()
        _service(live=chain).live(["P1", "P1"])
        assert chain.requested == [["P1"]]


class TestCatalogue:
    def test_the_catalogue_lists_what_may_be_asked_for(self):
        assert _CATALOGUE.leagues == ("P1", "E0", "CL")

    def test_a_league_is_recognised_case_sensitively(self):
        """League codes are identifiers, not free text: "p1" is not "P1"."""
        with pytest.raises(UnknownLeagueError):
            _service().live(["p1"])
