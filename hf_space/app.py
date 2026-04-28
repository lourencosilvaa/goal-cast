"""HuggingFace Space — Football Prediction FastAPI service.

Loads the trained model once at startup, then serves on-demand predictions.
Accepts POST /infer with {date, league_codes} and returns predictions.
"""

import csv
import io
import os
import time as _time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.config_loader import SpaceConfig, load_config
from src.models.data_loader import FootballDataLoader
from src.models.feature_engineer import FeatureEngineer
from src.models.predictor import MatchPredictor

_PREDICTOR: MatchPredictor | None = None
_CONFIG: SpaceConfig | None = None

_FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"
_FIXTURES_CACHE = Path("/tmp/fixtures.csv")
_FIXTURES_CACHE_AGE = 60 * 60  # 1 hour

DIVISION_MAP = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "P1": "Liga Portugal",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _PREDICTOR, _CONFIG
    _CONFIG = load_config()
    model_dir = _download_model(_CONFIG)
    _PREDICTOR = MatchPredictor(model_dir)
    yield


app = FastAPI(title="Football Prediction API", lifespan=lifespan)


def _download_model(config: SpaceConfig) -> Path:
    from huggingface_hub import snapshot_download

    repo_id = config.huggingface.repo_id
    hf_token = config.huggingface.hf_token or None
    local_dir = Path(config.huggingface.local_dir)
    path = snapshot_download(repo_id=repo_id, token=hf_token, local_dir=local_dir)
    return Path(path)


class InferRequest(BaseModel):
    date: str | None = None
    league_codes: list[str] | None = None


class InferResponse(BaseModel):
    predictions: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _PREDICTOR is not None}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    if _PREDICTOR is None or _CONFIG is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    target_date = req.date or datetime.now().strftime("%d/%m/%Y")
    fixtures = _fetch_fixtures(target_date, req.league_codes)
    if not fixtures:
        return InferResponse(predictions=[])

    historical_df = _load_historical_data(_CONFIG)
    enriched = _engineer_features(historical_df, _CONFIG)

    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        features = _build_features(fixture, enriched)
        feature_df = pd.DataFrame([features])
        try:
            preds = _PREDICTOR.predict(feature_df)
            if preds:
                d = preds[0].to_dict()
                d["league"] = fixture["league"]
                d["match_date"] = fixture["date"]
                d["time"] = fixture["time"]
                results.append(d)
        except Exception:
            continue

    return InferResponse(predictions=results)


def _fetch_fixtures(
    target_date: str, league_codes: list[str] | None
) -> list[dict[str, str]]:
    if _FIXTURES_CACHE.exists():
        age = _time.time() - _FIXTURES_CACHE.stat().st_mtime
        if age < _FIXTURES_CACHE_AGE:
            text = _FIXTURES_CACHE.read_text(encoding="utf-8-sig")
        else:
            text = _download_fixtures_csv()
    else:
        text = _download_fixtures_csv()

    reader = csv.DictReader(io.StringIO(text))
    fixtures: list[dict[str, str]] = []
    for row in reader:
        if row.get("Date", "") != target_date:
            continue
        div = row.get("Div", "")
        if league_codes and div not in league_codes:
            continue
        fixtures.append(
            {
                "division": div,
                "league": DIVISION_MAP.get(div, div),
                "date": row.get("Date", ""),
                "time": row.get("Time", ""),
                "home_team": row.get("HomeTeam", ""),
                "away_team": row.get("AwayTeam", ""),
            }
        )
    return fixtures


def _download_fixtures_csv() -> str:
    resp = requests.get(_FIXTURES_CSV_URL, timeout=30)
    resp.raise_for_status()
    text = resp.content.decode("utf-8-sig")
    _FIXTURES_CACHE.write_text(text, encoding="utf-8")
    return text


def _load_historical_data(config: SpaceConfig) -> pd.DataFrame:
    try:
        loader = FootballDataLoader(config.data, config.huggingface)
        return loader.load_all()
    except Exception:
        return pd.DataFrame()


def _engineer_features(df: pd.DataFrame, config: SpaceConfig) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        fe = FeatureEngineer(config.features)
        return fe.build_all_features(df)
    except Exception:
        return pd.DataFrame()


def _build_features(
    fixture: dict[str, str], enriched: pd.DataFrame
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "HomeTeam": fixture["home_team"],
        "AwayTeam": fixture["away_team"],
    }
    if not enriched.empty:
        home_rows = enriched[enriched["HomeTeam"] == fixture["home_team"]]
        if not home_rows.empty:
            last = home_rows.iloc[-1]
            for col in enriched.columns:
                if col not in ("HomeTeam", "AwayTeam", "FTR", "Date"):
                    try:
                        features[col] = float(last[col])
                    except (TypeError, ValueError):
                        pass
    return features
