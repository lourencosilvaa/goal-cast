"""Fall back across results providers until one answers.

The contract is copied verbatim from :mod:`src.scrapers.european.chained`,
including its sharp edge: the **first non-empty** answer wins rather than a
merge of every provider's. Merging would need cross-provider deduplication
over team names that are deliberately not yet canonical ("Porto" here,
"FC Porto" there), so it would have to guess exactly where this project
refuses to.

The cost is that a provider returning a partial list is taken as complete,
which makes ordering a correctness property rather than a preference: the
source with the best coverage goes first.

``last_source`` is not diagnostics decoration. An empty result with a source
means "that provider says nothing is happening"; an empty result with no
source means "nobody could tell us" — and those two call for opposite
responses from the layer above.
"""

from typing import ClassVar, Sequence

from src.scrapers.results.models import HistoryQuery, MatchResult


class _Chain:
    """Ordered fallback shared by the history and live chains."""

    name: ClassVar[str] = "chained"
    #: What the chain is fetching, for readable failure reports.
    subject: ClassVar[str] = "results"

    def __init__(self, providers: Sequence[object]) -> None:
        self._providers = list(providers)
        #: Which provider actually answered, or ``None`` if none did.
        self.last_source: str | None = None

    @property
    def providers(self) -> list[object]:
        """The chain as assembled, in order. Read-only: reordering is a
        configuration change, not something a caller does at runtime."""
        return list(self._providers)

    def _run(self, call: str, *args: object) -> list[MatchResult]:
        # Cleared first: leaving the previous call's winner in place would
        # attribute today's empty answer to yesterday's provider.
        self.last_source = None
        for provider in self._providers:
            name = getattr(provider, "name", provider.__class__.__name__)
            try:
                matches = list(getattr(provider, call)(*args))
            except Exception as exc:
                # Reported, not swallowed: a silent chain is indistinguishable
                # from a genuinely empty result.
                print(
                    f"[results:{self.subject}] {name} failed ({exc}) — "
                    f"trying the next source"
                )
                continue
            if matches:
                self.last_source = name
                return matches
        return []


class ChainedHistoryProvider(_Chain):
    """Queries history providers in order, first non-empty answer wins."""

    name: ClassVar[str] = "chained-history"
    subject: ClassVar[str] = "history"

    def fetch_history(self, query: HistoryQuery) -> list[MatchResult]:
        return self._run("fetch_history", query)


class ChainedLiveProvider(_Chain):
    """Queries live providers in order, first non-empty answer wins."""

    name: ClassVar[str] = "chained-live"
    subject: ClassVar[str] = "live"

    def fetch_live(self, leagues: list[str]) -> list[MatchResult]:
        return self._run("fetch_live", list(leagues))
