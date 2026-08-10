"""On-demand live polling with a TTL, and the diff that produces events.

**Why not a background poller.** A loop refreshing every minute would spend
the provider's quota around the clock whether or not anyone is looking, and it
would need asyncio or a thread — which §4.4 says to ask about before
introducing, and which multiplies the surface a test has to pin down. With a
60-second TTL the observable behaviour is identical to a continuous poller for
anyone actually watching: the first request after the window refreshes, and
everyone else is served the same snapshot.

**The TTL gates attempts, not the age of what is served.** The distinction
only shows up when things go wrong, which is when it matters: if the window
were measured against the snapshot's own timestamp, a provider that has
started failing would be re-queried on *every* request — hammering a service
that is already struggling, and burning the ten-per-minute allowance on
failures. Attempt-gating keeps the request rate flat regardless.

There is deliberately no forced-refresh parameter. A refresh button that
bypassed the TTL would hand any user a way to exhaust the quota, and inside a
60-second window there is nothing new to fetch.
"""

from datetime import datetime
from typing import Callable, ClassVar, Protocol, Sequence

from config.config_loader import ResultsLiveConfig
from src.scrapers.results.elapsed import utc_now
from src.scrapers.results.models import (
    EventType,
    LiveSnapshot,
    LiveUpdate,
    MatchEvent,
    MatchResult,
    MatchStatus,
)


class SnapshotStore(Protocol):
    """Somewhere a snapshot outlives the process.

    A Protocol, not an import: the implementation lives in the service
    (:class:`src.results_service.repository.JsonResultsRepository`), and this
    package must not depend on the service that composes it.
    """

    def load(self, key: Sequence[str]) -> LiveSnapshot | None: ...

    def save(self, key: Sequence[str], snapshot: LiveSnapshot) -> None: ...


class _Entry:
    """One cached league-set: what was served, why, and when it was tried."""

    def __init__(
        self,
        snapshot: LiveSnapshot,
        events: tuple[MatchEvent, ...],
        last_attempt: datetime,
        diffable: bool = True,
    ) -> None:
        self.snapshot = snapshot
        self.events = events
        self.last_attempt = last_attempt
        #: Whether the next refresh may diff against this snapshot. False for
        #: one restored from disk: it was current before the process
        #: restarted, so the goals scored meanwhile are history, not news, and
        #: announcing them would replay a match that finished hours ago.
        self.diffable = diffable


class LiveResultsTracker:
    """Serves live snapshots, refreshing at most once per configured window."""

    name: ClassVar[str] = "live-tracker"

    def __init__(
        self,
        provider: object,
        config: ResultsLiveConfig,
        clock: Callable[[], datetime] | None = None,
        store: SnapshotStore | None = None,
    ) -> None:
        self._provider = provider
        self._config = config
        #: Injected so TTL and staleness are tested as policy, not as sleeps.
        self._clock = clock or utc_now
        #: Optional: without one the cache simply does not survive a restart.
        self._store = store
        self._cache: dict[tuple[str, ...], _Entry] = {}

    @property
    def provider(self) -> object:
        """The live chain behind this tracker, for wiring and diagnosis."""
        return self._provider

    def get_snapshot(self, leagues: list[str]) -> LiveUpdate:
        """The current view of ``leagues``, from cache or from the chain."""
        now = self._clock()
        key = self._key(leagues)
        entry = self._cache.get(key) or self._restore(key)

        if entry is None or self._window_elapsed(entry.last_attempt, now):
            entry = self._refresh(leagues, entry, now, key)
        self._cache[key] = entry

        return LiveUpdate(
            snapshot=entry.snapshot,
            events=entry.events,
            stale=self._is_stale(entry.snapshot.fetched_at, now),
        )

    # ── persistence ──────────────────────────────────────────────────────

    def _restore(self, key: tuple[str, ...]) -> _Entry | None:
        """Seed the cache from the store, if there is one and it has anything.

        ``last_attempt`` becomes the snapshot's own timestamp rather than now,
        so a restored snapshot is subject to exactly the same TTL as one this
        process fetched: old enough and the next request refreshes it, recent
        enough and it is served as-is.
        """
        if self._store is None:
            return None
        try:
            snapshot = self._store.load(key)
        except Exception as exc:
            print(f"[results:tracker] could not restore snapshot ({exc})")
            return None
        if snapshot is None:
            return None
        return _Entry(snapshot, (), snapshot.fetched_at, diffable=False)

    def _persist(self, key: tuple[str, ...], snapshot: LiveSnapshot) -> None:
        if self._store is None:
            return
        try:
            self._store.save(key, snapshot)
        except Exception as exc:
            # A cache that cannot be written is still a working tracker.
            print(f"[results:tracker] could not persist snapshot ({exc})")

    # ── refreshing ───────────────────────────────────────────────────────

    def _refresh(
        self,
        leagues: list[str],
        previous: _Entry | None,
        now: datetime,
        key: tuple[str, ...],
    ) -> _Entry:
        try:
            matches = list(self._provider.fetch_live(list(leagues)))  # type: ignore[attr-defined]
        except Exception as exc:
            # The chain already reports per-provider failures; this catches a
            # chain that itself broke. An exception must not reach the caller:
            # the last known scores are a better answer than a 500.
            print(f"[results:tracker] live fetch failed ({exc})")
            matches = []

        if not matches:
            # Cannot distinguish "nothing is being played" from "every source
            # is down", so the previous snapshot is kept and allowed to age
            # into `stale` rather than being replaced by an empty board that
            # asserts there are no matches.
            if previous is not None:
                previous.last_attempt = now
                return previous
            return _Entry(LiveSnapshot(now, (), ""), (), now)

        snapshot = LiveSnapshot(
            fetched_at=now,
            matches=tuple(matches),
            source=str(getattr(self._provider, "last_source", "") or ""),
        )
        earlier = previous.snapshot if previous and previous.diffable else None
        events = self._diff(earlier, snapshot, now)
        self._persist(key, snapshot)
        return _Entry(snapshot, events, now)

    # ── the diff ─────────────────────────────────────────────────────────

    @staticmethod
    def _diff(
        previous: LiveSnapshot | None, current: LiveSnapshot, now: datetime
    ) -> tuple[MatchEvent, ...]:
        """What changed between two snapshots.

        Only matches present in *both* can produce an event. A match seen for
        the first time already carries whatever score it had when we arrived,
        and reporting those as goals would announce a 3-0 the moment a page is
        opened — the first snapshot of a day would look like a flurry of
        scoring that happened hours ago.
        """
        if previous is None:
            return ()
        before = {match.key: match for match in previous.matches}
        events: list[MatchEvent] = []
        for match in current.matches:
            earlier = before.get(match.key)
            if earlier is None:
                continue
            for event_type in LiveResultsTracker._changes(earlier, match):
                events.append(MatchEvent(type=event_type, match=match, detected_at=now))
        return tuple(events)

    @staticmethod
    def _changes(earlier: MatchResult, later: MatchResult) -> list[EventType]:
        changes: list[EventType] = []
        if LiveResultsTracker._scored(earlier, later):
            changes.append(EventType.GOAL)
        if earlier.status is MatchStatus.SCHEDULED and later.status is MatchStatus.LIVE:
            changes.append(EventType.KICKOFF)
        if (
            earlier.status is not MatchStatus.FINISHED
            and later.status is MatchStatus.FINISHED
        ):
            changes.append(EventType.FULL_TIME)
        return changes

    @staticmethod
    def _scored(earlier: MatchResult, later: MatchResult) -> bool:
        """A goal is a score going *up*.

        A score going down is VAR rescinding one, which is not a goal — and
        announcing it as one would be a fiction the UI then has to unsay.
        """
        for before, after in (
            (earlier.home_goals, later.home_goals),
            (earlier.away_goals, later.away_goals),
        ):
            if before is not None and after is not None and after > before:
                return True
        return False

    # ── windows ──────────────────────────────────────────────────────────

    def _window_elapsed(self, last_attempt: datetime, now: datetime) -> bool:
        elapsed = (now - last_attempt).total_seconds()
        return elapsed >= self._config.poll_interval_seconds

    def _is_stale(self, fetched_at: datetime, now: datetime) -> bool:
        age = (now - fetched_at).total_seconds()
        return age >= self._config.stale_after_seconds

    @staticmethod
    def _key(leagues: list[str]) -> tuple[str, ...]:
        """Cache key. Sorted and de-duplicated so the same request written two
        ways shares one cache entry rather than costing two calls."""
        return tuple(sorted(set(leagues)))
