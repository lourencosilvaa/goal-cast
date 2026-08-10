"""The domain objects the results track passes around.

They are frozen because a snapshot is compared against its successor to derive
events; a mutable match would let one snapshot's contents change under the
diff and produce events that never happened.
"""

from datetime import datetime

import pytest

from src.scrapers.results.models import (
    EventType,
    HistoryQuery,
    LiveSnapshot,
    MatchEvent,
    MatchResult,
    MatchStatus,
)


def _match(**overrides) -> MatchResult:
    defaults = dict(
        league="P1",
        kickoff=datetime(2026, 8, 9, 17, 0),
        home_team="Porto",
        away_team="Alverca",
        status=MatchStatus.LIVE,
        home_goals=2,
        away_goals=0,
        minute="67'",
        source="football-data.org",
    )
    defaults.update(overrides)
    return MatchResult(**defaults)


class TestMatchResult:
    def test_a_match_is_immutable(self):
        with pytest.raises(Exception):
            _match().home_goals = 3  # type: ignore[misc]

    def test_a_scheduled_match_carries_no_goals(self):
        match = _match(status=MatchStatus.SCHEDULED, home_goals=None, away_goals=None)
        assert match.home_goals is None

    def test_the_key_identifies_the_same_fixture_across_snapshots(self):
        first = _match(home_goals=0, away_goals=0, minute="12'")
        later = _match(home_goals=2, away_goals=0, minute="67'")
        assert first.key == later.key

    def test_the_key_separates_the_same_teams_on_different_days(self):
        """Home and away meet twice a season; a reversed or later fixture must
        not be diffed against the earlier one."""
        first = _match(kickoff=datetime(2026, 8, 9, 17, 0))
        second = _match(kickoff=datetime(2026, 12, 20, 17, 0))
        assert first.key != second.key

    def test_the_key_separates_the_reverse_fixture(self):
        home = _match()
        away = _match(home_team="Alverca", away_team="Porto")
        assert home.key != away.key


class TestMatchStatus:
    def test_statuses_serialise_as_their_lowercase_names(self):
        assert MatchStatus.FINISHED.value == "finished"

    def test_status_is_a_string_enum_so_it_survives_json(self):
        assert MatchStatus.LIVE == "live"


class TestHistoryQuery:
    def test_a_query_carries_league_and_season(self):
        query = HistoryQuery(league="P1", season="2526")
        assert (query.league, query.season) == ("P1", "2526")


class TestLiveSnapshot:
    def test_a_snapshot_records_which_provider_answered(self):
        snapshot = LiveSnapshot(
            fetched_at=datetime(2026, 8, 9, 17, 30),
            matches=(_match(),),
            source="football-data.org",
        )
        assert snapshot.source == "football-data.org"

    def test_an_empty_snapshot_is_valid(self):
        snapshot = LiveSnapshot(
            fetched_at=datetime(2026, 8, 9, 3, 0), matches=(), source=""
        )
        assert snapshot.matches == ()


class TestMatchEvent:
    def test_an_event_points_at_the_match_it_describes(self):
        match = _match()
        event = MatchEvent(
            type=EventType.GOAL, match=match, detected_at=datetime(2026, 8, 9, 17, 30)
        )
        assert event.match is match

    def test_event_types_serialise_as_strings(self):
        assert EventType.FULL_TIME.value == "full_time"
