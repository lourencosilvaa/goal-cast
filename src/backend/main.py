"""
FastAPI backend for the Football Prediction Agent.

Provides prediction endpoints with in-memory caching.
Precomputes all league predictions in a background thread at startup.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backend.api import (
    admin,
    ai,
    evaluation,
    exports,
    keys,
    leagues,
    predictions,
    profile,
    status,
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
    print("Backend ready (predictions served from Supabase).")
    yield


app = FastAPI(title="Football Prediction Agent API", lifespan=lifespan)

_cors_origins = os.environ.get(
    "CORS_ORIGINS", "https://football-prediction-s79r.onrender.com"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router)
app.include_router(leagues.router)
app.include_router(ai.router)
app.include_router(exports.router)
app.include_router(evaluation.router)
app.include_router(keys.router)
app.include_router(admin.router)
app.include_router(profile.router)
app.include_router(status.router)
