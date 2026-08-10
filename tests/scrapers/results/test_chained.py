"""Falling back across results providers.

Same contract as the fixtures chain (``src/scrapers/european/chained.py``):
first non-empty answer wins, a provider that raises is skipped rather than
allowed to end the chain, and which source answered is recorded — because
"nothing today" and "every source is down" look identical from the outside and
call for opposite responses.
"""

from datetime import datetime

from src.scrapers.results.chained import ChainedHistoryProvider, ChainedLiveProvider
from src.scrapers.results.models import HistoryQuery, MatchResult, MatchStatus

_QUERY = HistoryQuery(league="P1", season="2526")


def _match(home: str) -> MatchResult:
    return MatchResult(
        league="P1",
        kickoff=datetime(2026, 8, 9, 17, 0),
        home_team=home,
        away_team="Alverca",
        status=MatchStatus.FINISHED,
        home_goals=1,
        away_goals=0,
        source="stub",
    )


class _HistoryStub:
    def __init__(self, name, matches=None, error=None, enabled=True):
        self.name = name
        self._matches = matches or []
        self._error = error
        self.enabled = enabled
        self.called = False

    def fetch_history(self, query):
        self.called = True
        if self._error:
            raise self._error
        return list(self._matches)


class _LiveStub(_HistoryStub):
    def fetch_live(self, leagues):
        self.called = True
        if self._error:
            raise self._error
        return list(self._matches)


class TestHistoryChain:
    def test_the_first_provider_with_results_wins(self):
        chain = ChainedHistoryProvider(
            [_HistoryStub("corpus", [_match("Porto")]), _HistoryStub("api")]
        )
        assert [m.home_team for m in chain.fetch_history(_QUERY)] == ["Porto"]

    def test_later_providers_are_not_called_once_one_answers(self):
        second = _HistoryStub("api", [_match("Benfica")])
        ChainedHistoryProvider(
            [_HistoryStub("corpus", [_match("Porto")]), second]
        ).fetch_history(_QUERY)
        assert not second.called

    def test_an_empty_provider_hands_over_to_the_next(self):
        chain = ChainedHistoryProvider(
            [_HistoryStub("corpus"), _HistoryStub("api", [_match("Benfica")])]
        )
        assert [m.home_team for m in chain.fetch_history(_QUERY)] == ["Benfica"]

    def test_a_raising_provider_is_skipped_rather_than_ending_the_chain(self):
        chain = ChainedHistoryProvider(
            [
                _HistoryStub("corpus", error=RuntimeError("disk on fire")),
                _HistoryStub("api", [_match("Benfica")]),
            ]
        )
        assert [m.home_team for m in chain.fetch_history(_QUERY)] == ["Benfica"]

    def test_the_answering_provider_is_recorded(self):
        chain = ChainedHistoryProvider(
            [_HistoryStub("corpus"), _HistoryStub("api", [_match("Benfica")])]
        )
        chain.fetch_history(_QUERY)
        assert chain.last_source == "api"

    def test_no_source_is_recorded_when_nobody_answers(self):
        chain = ChainedHistoryProvider([_HistoryStub("corpus"), _HistoryStub("api")])
        assert chain.fetch_history(_QUERY) == []
        assert chain.last_source is None

    def test_a_stale_source_from_a_previous_call_is_cleared(self):
        """Otherwise an empty answer would be attributed to whoever answered
        last time."""
        answering = _HistoryStub("api", [_match("Benfica")])
        chain = ChainedHistoryProvider([answering])
        chain.fetch_history(_QUERY)
        answering._matches = []
        chain.fetch_history(_QUERY)
        assert chain.last_source is None

    def test_an_empty_chain_answers_empty(self):
        chain = ChainedHistoryProvider([])
        assert chain.fetch_history(_QUERY) == []


class TestLiveChain:
    def test_the_first_provider_with_results_wins(self):
        chain = ChainedLiveProvider(
            [_LiveStub("api", [_match("Porto")]), _LiveStub("flashscore")]
        )
        assert [m.home_team for m in chain.fetch_live(["P1"])] == ["Porto"]

    def test_the_fallback_is_used_when_the_api_is_empty(self):
        chain = ChainedLiveProvider(
            [_LiveStub("api"), _LiveStub("flashscore", [_match("Porto")])]
        )
        assert chain.fetch_live(["P1"]) != []
        assert chain.last_source == "flashscore"

    def test_the_fallback_is_used_when_the_api_raises(self):
        chain = ChainedLiveProvider(
            [
                _LiveStub("api", error=RuntimeError("429")),
                _LiveStub("flashscore", [_match("Porto")]),
            ]
        )
        assert [m.home_team for m in chain.fetch_live(["P1"])] == ["Porto"]
        assert chain.last_source == "flashscore"

    def test_the_requested_leagues_reach_every_provider_tried(self):
        class _Recorder(_LiveStub):
            def fetch_live(self, leagues):
                self.leagues = list(leagues)
                return super().fetch_live(leagues)

        second = _Recorder("flashscore", [_match("Porto")])
        ChainedLiveProvider([_Recorder("api"), second]).fetch_live(["P1", "E0"])
        assert second.leagues == ["P1", "E0"]
