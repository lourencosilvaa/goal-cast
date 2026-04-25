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
    """Pre-load the ML model, config, and warm the cache on startup."""
    import os
    from pathlib import Path

    from config.config_loader import load_config
    from src.backend.services.model_loader import ModelLoader
    from src.backend.services.prediction_service import PredictionService

    config = load_config()

    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo = os.environ.get("HF_REPO_ID", "")
    if hf_token and hf_repo:
        loader = ModelLoader(
            repo_id=hf_repo,
            hf_token=hf_token,
            local_dir=Path(os.environ.get("HF_LOCAL_DIR", "/tmp/hf_models")),
        )
        try:
            downloaded_path = loader.download()
            model_path = loader.get_model_path("ensemble_model.joblib")
            config.output.models_dir = str(model_path.parent)
            config.huggingface.local_dir = str(downloaded_path)
            print(f"HF model + datasets downloaded to {downloaded_path}")
        except RuntimeError as e:
            print(f"HF model download skipped (non-fatal): {e}")

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
app.include_router(keys.router)
app.include_router(admin.router)
app.include_router(profile.router)
app.include_router(status.router)
