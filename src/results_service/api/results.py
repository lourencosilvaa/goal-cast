"""``GET /live`` and ``GET /history``.

Thin on purpose: parse the query string, call the service, translate its
domain errors into status codes, and map internal matches onto the shared wire
contract. No provider knowledge and no policy live here.

The error translation is the part worth reading. Three outcomes are kept
apart, because collapsing any two of them loses information the caller needs:

* **422** — the request named a league or season this deployment does not
  offer. The caller's fault, fixable, and explicitly *not* an empty list.
* **502** — the service broke while trying. Never a 200 with no matches: the
  app must be able to tell "nothing is being played" from "we could not find
  out", and only one of those should reach a user as an empty board.
* **200 with ``stale: true``** — we have an answer, but it is old.
"""

from datetime import datetime
from typing import Annotated, Callable, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from src.contracts.results import (
    EventModel,
    HistoryResponse,
    LiveResultsResponse,
    MatchModel,
)
from src.results_service.auth import require_api_key
from src.results_service.service import (
    HistoryResult,
    ResultsService,
    UnknownLeagueError,
    UnknownSeasonError,
)
from src.scrapers.results.elapsed import estimate_elapsed_minutes, utc_now
from src.scrapers.results.models import (
    HistoryQuery,
    MatchEvent,
    MatchResult,
    MatchStatus,
)

#: Statuses whose minute is worked out from the clock rather than observed or
#: fixed by the status itself.
_CLOCK_ESTIMATED_STATUSES = (MatchStatus.LIVE,)

router = APIRouter(tags=["results"], dependencies=[Depends(require_api_key)])

#: Whatever the guarded call returns, so the endpoints keep their own types.
_Payload = TypeVar("_Payload")


def get_service(request: Request) -> ResultsService:
    """The service built at boot, held on the app rather than in a global."""
    service: ResultsService = request.app.state.results_service
    return service


@router.get("/live", response_model=LiveResultsResponse)
def live(
    service: Annotated[ResultsService, Depends(get_service)],
    leagues: str = Query(
        default="",
        description=(
            "Comma-separated league codes. Omit for every offered league — "
            "one upstream request covers them all anyway."
        ),
    ),
) -> LiveResultsResponse:
    update = _guard(lambda: service.live(_split(leagues)))
    return LiveResultsResponse(
        fetched_at=update.snapshot.fetched_at,
        source=update.snapshot.source,
        stale=update.stale,
        matches=[_match(m) for m in update.snapshot.matches],
        events=[_event(e) for e in update.events],
    )


@router.get("/history", response_model=HistoryResponse)
def history(
    service: Annotated[ResultsService, Depends(get_service)],
    league: Annotated[str, Query(description="League code, e.g. P1")],
    season: Annotated[str, Query(description="Season in YYZZ form, e.g. 2526")],
) -> HistoryResponse:
    result = _guard(lambda: service.history(HistoryQuery(league=league, season=season)))
    return _history_response(result)


# ── translation ──────────────────────────────────────────────────────────


def _guard(call: Callable[[], _Payload]) -> _Payload:
    """Run ``call``, turning domain errors into the right status codes."""
    try:
        return call()
    except (UnknownLeagueError, UnknownSeasonError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        # Deliberately 502 rather than 500: from the caller's side this
        # service is a gateway to the providers, and the distinction is what
        # tells the app to say "results unavailable" instead of "no matches".
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"results providers failed: {exc}",
        ) from exc


def _split(raw: str) -> list[str]:
    """``"E0, P1,"`` → ``["E0", "P1"]``.

    Blanks are dropped rather than passed on: a trailing comma is a typo, and
    an empty code would be refused as an unknown league — a 422 for what the
    caller plainly meant.
    """
    return [code.strip() for code in raw.split(",") if code.strip()]


def _match(match: MatchResult, now: datetime | None = None) -> MatchModel:
    """One match on the wire, with the clock resolved.

    ``elapsed_minutes`` is computed here rather than in a provider because it
    depends on *when the answer is served*, not on when it was fetched — a
    snapshot held for the length of the TTL would otherwise report a minute up
    to a minute stale, and a restored one far worse.
    """
    moment = now or utc_now()
    return MatchModel(
        league=match.league,
        kickoff=match.kickoff,
        home_team=match.home_team,
        away_team=match.away_team,
        status=match.status.value,
        home_goals=match.home_goals,
        away_goals=match.away_goals,
        minute=match.minute,
        elapsed_minutes=estimate_elapsed_minutes(match, moment),
        # An empty provider clock is what makes it an estimate; the status
        # shortcuts (finished, half-time) are facts, not arithmetic.
        minute_estimated=not match.minute and match.status in _CLOCK_ESTIMATED_STATUSES,
        source=match.source,
        source_id=match.source_id,
    )


def _event(event: MatchEvent) -> EventModel:
    return EventModel(
        type=event.type.value,
        match=_match(event.match),
        detected_at=event.detected_at,
    )


def _history_response(result: HistoryResult) -> HistoryResponse:
    return HistoryResponse(
        league=result.league,
        season=result.season,
        source=result.source,
        matches=[_match(m) for m in result.matches],
    )
