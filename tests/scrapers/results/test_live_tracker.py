"""On-demand polling with a TTL, and the diff that turns snapshots into events.

"Real time" here is not a background loop. The TTL gates *attempts*, not the
age of what is served: with a dead provider, gating on snapshot age would
re-poll on every request and burn the 10/minute allowance on failures.

The clock is injected everywhere. A tracker tested with ``time.sleep`` would
be both slow and flaky, and would test the sleep rather than the policy.
"""

from datetime import datetime, timedelta

from config.config_loader import ResultsLiveConfig
from src.scrapers.results.live_tracker import LiveResultsTracker
from src.scrapers.results.models import (
    EventType,
    LiveSnapshot,
    MatchResult,
    MatchStatus,
)

_START = datetime(2026, 8, 9, 17, 0)


class _Clock:
    def __init__(self, now: datetime = _START):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now = self.now + timedelta(seconds=seconds)


class _Chain:
    """A live chain whose answer the test controls call by call."""

    def __init__(self, *answers):
        self._answers = list(answers)
        self.calls = 0
        self.last_source = "football-data.org"

    def fetch_live(self, leagues):
        self.calls += 1
        if not self._answers:
            return []
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        if isinstance(answer, Exception):
            raise answer
        if not answer:
            self.last_source = None
        return list(answer)


def _match(
    home_goals=0, away_goals=0, status=MatchStatus.LIVE, home="Porto", minute=""
) -> MatchResult:
    return MatchResult(
        league="P1",
        kickoff=_START,
        home_team=home,
        away_team="Alverca",
        status=status,
        home_goals=home_goals,
        away_goals=away_goals,
        minute=minute,
        source="football-data.org",
    )


def _tracker(chain, clock, store=None, **overrides) -> LiveResultsTracker:
    settings = dict(poll_interval_seconds=60, stale_after_seconds=300)
    settings.update(overrides)
    return LiveResultsTracker(
        chain, ResultsLiveConfig(**settings), clock=clock, store=store
    )


class _Store:
    """An in-memory stand-in for the service's snapshot repository."""

    def __init__(self, seeded=None):
        self.saved: list = []
        self._seeded = seeded or {}

    def load(self, key):
        return self._seeded.get(key)

    def save(self, key, snapshot):
        self.saved.append((key, snapshot))


class TestTimeToLive:
    def test_the_first_call_queries_the_chain(self):
        chain = _Chain([_match()])
        _tracker(chain, _Clock()).get_snapshot(["P1"])
        assert chain.calls == 1

    def test_a_second_call_inside_the_ttl_is_served_from_cache(self):
        chain = _Chain([_match()])
        clock = _Clock()
        tracker = _tracker(chain, clock)
        tracker.get_snapshot(["P1"])
        clock.advance(30)
        tracker.get_snapshot(["P1"])
        assert chain.calls == 1

    def test_a_call_after_the_ttl_refreshes(self):
        chain = _Chain([_match()])
        clock = _Clock()
        tracker = _tracker(chain, clock)
        tracker.get_snapshot(["P1"])
        clock.advance(61)
        tracker.get_snapshot(["P1"])
        assert chain.calls == 2

    def test_a_different_set_of_leagues_is_cached_separately(self):
        chain = _Chain([_match()])
        tracker = _tracker(chain, _Clock())
        tracker.get_snapshot(["P1"])
        tracker.get_snapshot(["P1", "E0"])
        assert chain.calls == 2

    def test_the_same_leagues_in_a_different_order_reuse_the_cache(self):
        chain = _Chain([_match()])
        tracker = _tracker(chain, _Clock())
        tracker.get_snapshot(["P1", "E0"])
        tracker.get_snapshot(["E0", "P1"])
        assert chain.calls == 1

    def test_a_failing_chain_is_not_retried_inside_the_ttl(self):
        """Quota protection matters most exactly when things are going wrong."""
        chain = _Chain([])
        clock = _Clock()
        tracker = _tracker(chain, clock)
        tracker.get_snapshot(["P1"])
        clock.advance(10)
        tracker.get_snapshot(["P1"])
        assert chain.calls == 1


class TestStaleness:
    def test_a_fresh_snapshot_is_not_stale(self):
        update = _tracker(_Chain([_match()]), _Clock()).get_snapshot(["P1"])
        assert update.stale is False

    def test_a_snapshot_older_than_the_stale_window_is_flagged(self):
        chain = _Chain([_match()], [])
        clock = _Clock()
        tracker = _tracker(chain, clock)
        tracker.get_snapshot(["P1"])
        clock.advance(301)
        update = tracker.get_snapshot(["P1"])
        assert update.stale is True

    def test_a_stale_snapshot_still_serves_its_last_known_matches(self):
        """Blanking the board would say "no matches" when the truth is "we
        cannot currently tell"."""
        chain = _Chain([_match(home_goals=2)], [])
        clock = _Clock()
        tracker = _tracker(chain, clock)
        tracker.get_snapshot(["P1"])
        clock.advance(301)
        update = tracker.get_snapshot(["P1"])
        assert [m.home_goals for m in update.snapshot.matches] == [2]

    def test_a_successful_refresh_clears_staleness(self):
        chain = _Chain([_match()], [], [_match(home_goals=1)])
        clock = _Clock()
        tracker = _tracker(chain, clock)
        tracker.get_snapshot(["P1"])
        clock.advance(301)
        tracker.get_snapshot(["P1"])
        clock.advance(61)
        assert tracker.get_snapshot(["P1"]).stale is False

    def test_an_empty_first_answer_is_not_stale(self):
        """Nothing is being played; that is current information, not staleness."""
        update = _tracker(_Chain([]), _Clock()).get_snapshot(["P1"])
        assert update.stale is False
        assert update.snapshot.matches == ()

    def test_a_raising_chain_yields_an_empty_snapshot_rather_than_an_error(self):
        update = _tracker(_Chain(RuntimeError("boom")), _Clock()).get_snapshot(["P1"])
        assert update.snapshot.matches == ()


class TestEvents:
    def _two_snapshots(self, first, second):
        chain = _Chain([first], [second])
        clock = _Clock()
        tracker = _tracker(chain, clock)
        tracker.get_snapshot(["P1"])
        clock.advance(61)
        return tracker.get_snapshot(["P1"])

    def test_the_first_snapshot_reports_no_events(self):
        """Every match in it would otherwise look like it just happened."""
        update = _tracker(_Chain([_match(home_goals=3)]), _Clock()).get_snapshot(["P1"])
        assert update.events == ()

    def test_a_home_goal_is_detected(self):
        update = self._two_snapshots(_match(0, 0), _match(1, 0))
        assert [e.type for e in update.events] == [EventType.GOAL]

    def test_an_away_goal_is_detected(self):
        update = self._two_snapshots(_match(1, 0), _match(1, 1))
        assert [e.type for e in update.events] == [EventType.GOAL]

    def test_a_goal_event_carries_the_updated_match(self):
        update = self._two_snapshots(_match(0, 0), _match(2, 0))
        assert update.events[0].match.home_goals == 2

    def test_kick_off_is_detected(self):
        update = self._two_snapshots(
            _match(None, None, MatchStatus.SCHEDULED), _match(0, 0, MatchStatus.LIVE)
        )
        assert [e.type for e in update.events] == [EventType.KICKOFF]

    def test_full_time_is_detected(self):
        update = self._two_snapshots(
            _match(1, 0, MatchStatus.LIVE), _match(1, 0, MatchStatus.FINISHED)
        )
        assert [e.type for e in update.events] == [EventType.FULL_TIME]

    def test_an_unchanged_match_produces_nothing(self):
        update = self._two_snapshots(_match(1, 0), _match(1, 0))
        assert update.events == ()

    def test_a_match_appearing_for_the_first_time_produces_no_event(self):
        update = self._two_snapshots(_match(home="Porto"), _match(home="Benfica"))
        assert update.events == ()

    def test_a_goal_and_full_time_in_one_refresh_are_both_reported(self):
        update = self._two_snapshots(
            _match(1, 0, MatchStatus.LIVE), _match(2, 0, MatchStatus.FINISHED)
        )
        assert {e.type for e in update.events} == {EventType.GOAL, EventType.FULL_TIME}

    def test_events_stay_attached_to_the_snapshot_they_describe(self):
        """A second viewer inside the TTL must see the same events, not none."""
        chain = _Chain([_match(0, 0)], [_match(1, 0)])
        clock = _Clock()
        tracker = _tracker(chain, clock)
        tracker.get_snapshot(["P1"])
        clock.advance(61)
        tracker.get_snapshot(["P1"])
        clock.advance(10)
        assert [e.type for e in tracker.get_snapshot(["P1"]).events] == [EventType.GOAL]

    def test_events_carry_the_time_they_were_detected(self):
        chain = _Chain([_match(0, 0)], [_match(1, 0)])
        clock = _Clock()
        tracker = _tracker(chain, clock)
        tracker.get_snapshot(["P1"])
        clock.advance(61)
        update = tracker.get_snapshot(["P1"])
        assert update.events[0].detected_at == clock.now

    def test_a_goal_disallowed_by_var_lowers_the_score_without_an_event(self):
        """A rescinded goal is not a goal; reporting one would be a fiction."""
        update = self._two_snapshots(_match(1, 0), _match(0, 0))
        assert update.events == ()


class TestPersistence:
    """The cache survives a restart when a store is injected.

    Free-tier hosting sleeps an idle service, so the in-memory cache is empty
    on the first request after a wake — exactly when a provider outage would
    otherwise leave the board blank with nothing to fall back on.
    """

    def test_a_refresh_is_persisted(self):
        store = _Store()
        _tracker(_Chain([_match(home_goals=1)]), _Clock(), store).get_snapshot(["P1"])
        assert [key for key, _ in store.saved] == [("P1",)]

    def test_a_cache_hit_is_not_persisted_again(self):
        store = _Store()
        clock = _Clock()
        tracker = _tracker(_Chain([_match()]), clock, store)
        tracker.get_snapshot(["P1"])
        clock.advance(30)
        tracker.get_snapshot(["P1"])
        assert len(store.saved) == 1

    def test_a_stored_snapshot_is_served_when_the_chain_answers_nothing(self):
        stored = LiveSnapshot(_START, (_match(home_goals=3),), "football-data.org")
        clock = _Clock(_START + timedelta(seconds=30))
        update = _tracker(_Chain([]), clock, _Store({("P1",): stored})).get_snapshot(
            ["P1"]
        )
        assert [m.home_goals for m in update.snapshot.matches] == [3]

    def test_a_restored_snapshot_ages_into_staleness_like_any_other(self):
        stored = LiveSnapshot(_START, (_match(),), "football-data.org")
        clock = _Clock(_START + timedelta(seconds=400))
        update = _tracker(_Chain([]), clock, _Store({("P1",): stored})).get_snapshot(
            ["P1"]
        )
        assert update.stale is True

    def test_a_restored_snapshot_does_not_suppress_a_fresh_fetch(self):
        stored = LiveSnapshot(_START, (_match(home_goals=3),), "football-data.org")
        chain = _Chain([_match(home_goals=1)])
        clock = _Clock(_START + timedelta(seconds=400))
        update = _tracker(chain, clock, _Store({("P1",): stored})).get_snapshot(["P1"])
        assert [m.home_goals for m in update.snapshot.matches] == [1]

    def test_a_restored_snapshot_produces_no_phantom_events(self):
        """Its scores were current hours ago; diffing against them would
        announce goals scored while the service was asleep."""
        stored = LiveSnapshot(_START, (_match(home_goals=0),), "football-data.org")
        chain = _Chain([_match(home_goals=3)])
        clock = _Clock(_START + timedelta(seconds=400))
        update = _tracker(chain, clock, _Store({("P1",): stored})).get_snapshot(["P1"])
        assert update.events == ()

    def test_a_failing_store_does_not_fail_the_request(self):
        class _Broken(_Store):
            def load(self, key):
                raise OSError("disk gone")

            def save(self, key, snapshot):
                raise OSError("disk gone")

        update = _tracker(_Chain([_match()]), _Clock(), _Broken()).get_snapshot(["P1"])
        assert len(update.snapshot.matches) == 1

    def test_no_store_is_a_valid_configuration(self):
        update = _tracker(_Chain([_match()]), _Clock()).get_snapshot(["P1"])
        assert len(update.snapshot.matches) == 1


class TestSnapshotContents:
    def test_the_snapshot_records_which_provider_answered(self):
        update = _tracker(_Chain([_match()]), _Clock()).get_snapshot(["P1"])
        assert update.snapshot.source == "football-data.org"

    def test_the_snapshot_records_when_it_was_taken(self):
        clock = _Clock()
        update = _tracker(_Chain([_match()]), clock).get_snapshot(["P1"])
        assert update.snapshot.fetched_at == _START
