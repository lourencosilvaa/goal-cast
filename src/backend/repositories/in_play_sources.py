"""What the in-play board reads from, expressed in the app's own services.

:mod:`src.backend.services.in_play_board` is written against two deliberately
narrow interfaces so that joining live scores to stored predictions does not
drag Supabase and the alias registry into it. These are the implementations
that do the dragging, and keeping them here is what leaves the board testable
with two dictionaries.

Both wrap something that already exists — ``PredictionService`` and
:class:`src.teams.resolver.TeamNameResolver` — and neither adds behaviour of
its own beyond the translation.
"""

from typing import Any, Sequence

from src.backend.services.in_play_board import NameMatcher, PredictionSource
from src.teams.resolver import TeamNameQuery, TeamNameResolver


class PredictionServiceSource(PredictionSource):
    """The offline predictions already in Supabase, read through their service.

    Nothing is cached here: ``PredictionService`` holds a ten-minute cache of
    its own, and a poll every sixty seconds mostly hits it.
    """

    def __init__(self, prediction_service: Any) -> None:
        self._service = prediction_service

    def matches(self, league: str, match_date: str) -> Sequence[dict[str, Any]]:
        rows: Sequence[dict[str, Any]] = self._service.get_league_predictions(
            league, match_date
        ).matches
        return rows


class ResolverNameMatcher(NameMatcher):
    """Canonical names via the shared resolver, which refuses to guess.

    The refusal is the feature. A live board spells clubs its own way, and the
    close calls are the dangerous ones — "Sporting CP" is nearer to "Sp
    Lisbon" than to anything else in the Primeira Liga, but "Sporting Gijon"
    is nearer still to "Sporting". An unresolved name leaves the match without
    an in-play card, which is visibly incomplete; a guessed one puts a
    confident number on the wrong fixture.
    """

    def __init__(self, resolver: TeamNameResolver) -> None:
        self._resolver = resolver

    def canonical(self, league: str, raw_name: str) -> str | None:
        return self._resolver.resolve(
            TeamNameQuery(league_code=league, raw_name=raw_name)
        ).canonical
