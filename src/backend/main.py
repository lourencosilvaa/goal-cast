"""
FastAPI backend for the Football Prediction Agent.

Provides prediction endpoints with in-memory caching.
Precomputes all league predictions in a background thread at startup.
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backend.api import ai, evaluation, exports, leagues, predictions


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-load the ML model, config, and warm the cache on startup."""
    from config.config_loader import load_config
    from src.backend.services.prediction_service import PredictionService

    config = load_config()
    service = PredictionService(config)
    app.state.prediction_service = service
    app.state.config = config
    print("ML model loaded, backend ready.")

    # Warm the prediction cache in a background thread so the API
    # is responsive immediately while predictions are being computed.
    async def _warm_cache() -> None:
        try:
            await asyncio.to_thread(service.get_all_leagues_predictions)
            print("Prediction cache warmed for all leagues.")
        except Exception as e:
            print(f"Cache warm-up error (non-fatal): {e}")

    task = asyncio.create_task(_warm_cache())
    yield
    task.cancel()


app = FastAPI(title="Football Prediction Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router)
app.include_router(leagues.router)
app.include_router(ai.router)
app.include_router(exports.router)
app.include_router(evaluation.router)
