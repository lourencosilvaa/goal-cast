"""Fall back across fixture providers until one answers.

Mirrors :class:`src.teams.resolver.ChainedTeamAliasRepository` — try each
source in order, skip the ones that fail, and never let a dead source take the
pipeline with it.

First **non-empty** wins rather than merging every provider's answer. Merging
would need cross-provider deduplication over team names that deliberately are
not yet canonical ("FC Kairat" here, "Kairat Almaty" there), so it would have
to guess exactly where this project refuses to. It would also multiply calls
against quotas measured in hundreds per month.

The cost of first-wins is that a provider returning a partial list is taken as
complete, which makes ordering a correctness property, not a preference: the
provider with the best coverage goes first.
"""

from typing import Any, ClassVar, Sequence

from src.scrapers.european.providers import EuropeanFixture, FixtureWindow


class ChainedFixtureProvider:
    """Queries providers in order and returns the first non-empty result."""

    name: ClassVar[str] = "chained"

    def __init__(self, providers: Sequence[Any]) -> None:
        self._providers = list(providers)
        #: Which provider actually answered, for reporting and diagnosis.
        self.last_source: str | None = None

    def fetch(
        self, window: FixtureWindow, competitions: list[str]
    ) -> list[EuropeanFixture]:
        self.last_source = None
        for provider in self._providers:
            name = getattr(provider, "name", provider.__class__.__name__)
            try:
                fixtures: list[EuropeanFixture] = list(
                    provider.fetch(window, competitions)
                )
            except Exception as exc:
                # Reported, not swallowed: a silent chain is indistinguishable
                # from a genuinely empty fixture list, and the two call for
                # opposite responses.
                print(f"[fixtures] {name} failed ({exc}) — trying the next source")
                continue
            if fixtures:
                self.last_source = name
                print(f"[fixtures] {len(fixtures)} fixtures from {name}")
                return fixtures
            print(f"[fixtures] {name} returned nothing")
        return []
