"""Health probe — unauthenticated, and deliberately shallow.

Render calls this to decide whether the container is alive, and it cannot send
an API key, so this is the one route outside the guard. It reports nothing a
caller could not already infer from getting a response at all, plus the build
it is running.

It does **not** touch a provider. A health check that polled upstream would
spend metered quota on every probe and would mark this service unhealthy
whenever a third party was down — restarting a container that is working
perfectly.
"""

import os

from fastapi import APIRouter

from src.contracts.results import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        commit=os.environ.get("GIT_SHA", "unknown"),
        build_time=os.environ.get("BUILD_TIME", "unknown"),
    )
