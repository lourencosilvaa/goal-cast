"""Tests for chaining fixture providers.

The chain exists because no single free tier is dependable. Right now only one
of the two configured providers can return anything at all — football-data.org
is dormant until its 2026-27 season loads — so the chain's job is less "load
balance" than "keep working when a source goes quiet, and never pretend a
partial answer is a complete one".

That last part is the sharp edge. Returning the first *non-empty* result is
only safe if the order reflects coverage quality, and a provider that raises
must be skipped rather than allowed to end the chain.
"""

from datetime import date

from src.scrapers.european.chained import ChainedFixtureProvider
from src.scrapers.european.providers import EuropeanFixture, FixtureWindow

_WINDOW = FixtureWindow(start=date(2026, 8, 1), end=date(2026, 8, 31))


def _fixture(home: str, away: str = "Away") -> EuropeanFixture:
    from datetime import datetime

    return EuropeanFixture(
        competition="CL",
        kickoff=datetime(2026, 8, 11, 15, 0),
        home_team=home,
        away_team=away,
        source="stub",
    )


class _Stub:
    def __init__(self, name: str, fixtures=None, error: Exception | None = None) -> None:
        self.name = name
        self._fixtures = fixtures or []
        self._error = error
        self.called = False

    def fetch(self, window, competitions):
        self.called = True
        if self._error:
            raise self._error
        return list(self._fixtures)


class TestOrdering:
    def test_the_first_provider_with_results_wins(self):
        first = _Stub("odds", [_fixture("Kairat")])
        second = _Stub("football-data", [_fixture("Arsenal")])
        result = ChainedFixtureProvider([first, second]).fetch(_WINDOW, ["CL"])
        assert [f.home_team for f in result] == ["Kairat"]

    def test_later_providers_are_not_called_once_one_answers(self):
        """Quota is scarce; a satisfied request must not spend more of it."""
        first = _Stub("odds", [_fixture("Kairat")])
        second = _Stub("football-data", [_fixture("Arsenal")])
        ChainedFixtureProvider([first, second]).fetch(_WINDOW, ["CL"])
        assert not second.called

    def test_an_empty_provider_falls_through_to_the_next(self):
        first = _Stub("odds", [])
        second = _Stub("football-data", [_fixture("Arsenal")])
        result = ChainedFixtureProvider([first, second]).fetch(_WINDOW, ["CL"])
        assert [f.home_team for f in result] == ["Arsenal"]


class TestFailureHandling:
    def test_a_raising_provider_is_skipped(self):
        first = _Stub("odds", error=RuntimeError("connection reset"))
        second = _Stub("football-data", [_fixture("Arsenal")])
        result = ChainedFixtureProvider([first, second]).fetch(_WINDOW, ["CL"])
        assert [f.home_team for f in result] == ["Arsenal"]

    def test_all_providers_failing_yields_empty(self):
        providers = [
            _Stub("a", error=RuntimeError("down")),
            _Stub("b", error=RuntimeError("down")),
        ]
        assert ChainedFixtureProvider(providers).fetch(_WINDOW, ["CL"]) == []

    def test_a_failure_is_reported_not_swallowed_silently(self, capsys):
        """A quiet chain is indistinguishable from a quiet fixture list."""
        providers = [_Stub("odds", error=RuntimeError("boom"))]
        ChainedFixtureProvider(providers).fetch(_WINDOW, ["CL"])
        assert "odds" in capsys.readouterr().out

    def test_no_providers_at_all_yields_empty(self):
        assert ChainedFixtureProvider([]).fetch(_WINDOW, ["CL"]) == []


class TestAttribution:
    def test_the_answering_provider_is_recorded(self):
        chain = ChainedFixtureProvider([_Stub("odds", [_fixture("Kairat")])])
        chain.fetch(_WINDOW, ["CL"])
        assert chain.last_source == "odds"

    def test_no_source_when_nothing_answered(self):
        chain = ChainedFixtureProvider([_Stub("odds", [])])
        chain.fetch(_WINDOW, ["CL"])
        assert chain.last_source is None
