"""App factory for the results service.

Same shape as ``src/backend/main.py`` and much smaller: no Supabase client, no
model loading, no static files. Two routers, one guard, and one service object
built *before* the first request rather than lazily on it — a chain assembled
on first use would make the first user pay for a configuration mistake that
should have stopped the deployment.

Kept apart from :mod:`src.results_service.main`, which is the process entry
point and builds an app the moment it is imported. Separating them is what
lets a test construct an app with its own runtime without a live config file
or a real key in the environment.
"""

from fastapi import FastAPI

from src.results_service.api import health, results
from src.results_service.auth import ApiKeyGuard
from src.results_service.factory import ServiceRuntime

TITLE = "Football Results Service"


def create_app(runtime: ServiceRuntime) -> FastAPI:
    """Wire an already-assembled runtime into an application.

    Taking a runtime rather than a config path means everything that can fail
    — a missing key, an unbuildable chain, a disabled track — has already
    failed by the time an app exists.
    """
    app = FastAPI(title=TITLE)
    # ApiKeyGuard refuses an empty key, so this line is also the last place an
    # unauthenticated deployment is stopped.
    app.state.api_key_guard = ApiKeyGuard(runtime.api_key)
    app.state.results_service = runtime.service
    app.include_router(health.router)
    app.include_router(results.router)
    return app


__all__ = ["ServiceRuntime", "TITLE", "create_app"]
