"""
FastAPI backend for the Football Prediction Agent.

Provides prediction endpoints with in-memory caching.
Precomputes all league predictions in a background thread at startup.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backend.api import (
    admin,
    ai,
    evaluation,
    exports,
    inference,
    keys,
    leagues,
    predictions,
    profile,
    stats,
    status,
    teams,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize config and prediction service on startup.

    The ML model no longer runs here — predictions are pre-computed
    by a GitHub Actions job and stored in Supabase.
    """
    from config.config_loader import load_config
    from src.backend.services.prediction_service import PredictionService

    config = load_config()
    service = PredictionService(config)
    app.state.prediction_service = service
    app.state.config = config
    from src.backend.core import dev_auth

    if dev_auth.is_enabled():
        print(
            "⚠️  DEV_AUTH_BYPASS is ENABLED — login is bypassed as "
            f"'{dev_auth.dev_user_id()}'. NEVER enable this in production."
        )
    print("Backend ready (predictions served from Supabase).")
    yield


app = FastAPI(title="Football Prediction Agent API", lifespan=lifespan)

app.include_router(predictions.router)
app.include_router(inference.router)
app.include_router(leagues.router)
app.include_router(ai.router)
app.include_router(exports.router)
app.include_router(evaluation.router)
app.include_router(keys.router)
app.include_router(admin.router)
app.include_router(profile.router)
app.include_router(status.router)
app.include_router(teams.router)
app.include_router(stats.router)

_static_dir = Path(__file__).resolve().parents[2] / "static"
if _static_dir.exists():
    _assets_dir = _static_dir / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        candidate = _static_dir / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_static_dir / "index.html"))
