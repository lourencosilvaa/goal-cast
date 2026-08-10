"""``/api/results/*`` — a thin proxy in front of the results service.

The browser never calls the results service directly, and the two reasons are
both load-bearing: the shared service key would have to be shipped to the
client, and the user's own authentication would have to be duplicated in a
second service. So the backend keeps the authentication it already has, holds
the service key server-side, and forwards.

There is no caching, no merging and no reshaping here. The results service
already owns the TTL, and a second cache in front of it would only make
"how old is this?" unanswerable.
"""

from typing import Annotated, Callable, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from src.backend.core.auth import get_approved_user
from src.backend.services.results_gateway import (
    HttpResultsGateway,
    MissingGatewayConfigError,
    ResultsGateway,
    ResultsRequestRejected,
    ResultsServiceUnavailable,
)
from src.contracts.results import HistoryResponse, LiveResultsResponse

router = APIRouter(prefix="/api/results", tags=["results"])

#: Whatever the guarded call returns, so the endpoints keep their own types.
_Payload = TypeVar("_Payload")


def get_results_gateway(request: Request) -> ResultsGateway:
    """The configured gateway.

    Built per request rather than at start-up because it is a stateless
    wrapper around one HTTP call, and because reading the environment here
    means a variable added on the platform takes effect on the next request
    instead of the next deploy.
    """
    return HttpResultsGateway(request.app.state.config.results_gateway)


@router.get("/live", response_model=LiveResultsResponse)
def live(
    _: Annotated[str, Depends(get_approved_user)],
    gateway: Annotated[ResultsGateway, Depends(get_results_gateway)],
    leagues: str = Query(
        default="", description="Comma-separated league codes; omit for all."
    ),
) -> LiveResultsResponse:
    return guard_results_call(lambda: gateway.live(split_leagues(leagues)))


@router.get("/history", response_model=HistoryResponse)
def history(
    _: Annotated[str, Depends(get_approved_user)],
    gateway: Annotated[ResultsGateway, Depends(get_results_gateway)],
    league: Annotated[str, Query(description="League code, e.g. P1")],
    season: Annotated[str, Query(description="Season in YYZZ form, e.g. 2526")],
) -> HistoryResponse:
    return guard_results_call(lambda: gateway.history(league, season))


def guard_results_call(call: Callable[[], _Payload]) -> _Payload:
    """Translate gateway failures into status codes.

    Public because ``/api/in-play`` reads the same live board through the same
    gateway, and two routes that share a failure mode should not disagree
    about what it looks like from outside.

    503 for anything that means "the results service is not answering",
    including our own missing configuration — from a user's side that is an
    outage, and the detail still names the variable for the logs. 422 for a
    request the service refused, which retrying will not fix.
    """
    try:
        return call()
    except ResultsRequestRejected as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except (ResultsServiceUnavailable, MissingGatewayConfigError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


def split_leagues(raw: str) -> list[str]:
    """``"P1, E0"`` → ``["P1", "E0"]``; empty means every offered league."""
    return [code.strip() for code in raw.split(",") if code.strip()]
