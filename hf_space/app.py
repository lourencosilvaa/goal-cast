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
from src.models.data_cleaner import DataCleaner
from src.models.data_loader import FootballDataLoader
from src.models.elo import FootballELO
from src.models.feature_engineer import FeatureEngineer
from src.models.predictor import MatchPredictor

_PREDICTOR: MatchPredictor | None = None
_CONFIG: SpaceConfig | None = None
_ENRICHED_DATA: pd.DataFrame | None = None
_TEAMS_BY_LEAGUE: dict[str, list[str]] = {}

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
    global _PREDICTOR, _CONFIG, _ENRICHED_DATA, _TEAMS_BY_LEAGUE
    _CONFIG = load_config()
    model_dir = _download_model(_CONFIG)
    _PREDICTOR = MatchPredictor(model_dir)
    # Pre-compute enriched features once at startup
    historical_df = _load_historical_data(_CONFIG)
    # Extract teams per league from raw data before enrichment
    _TEAMS_BY_LEAGUE = _extract_teams_by_league(historical_df)
    _ENRICHED_DATA = _engineer_features(historical_df, _CONFIG)
    yield


app = FastAPI(title="Football Prediction API", lifespan=lifespan)


def _download_model(config: SpaceConfig) -> Path:
    from huggingface_hub import snapshot_download

    repo_id = config.huggingface.repo_id
    if not repo_id:
        raise RuntimeError(
            "HF_REPO_ID environment variable is not set. "
            "Set it in the HuggingFace Space secrets/environment variables."
        )
    hf_token = config.huggingface.hf_token or None
    local_dir = Path(config.huggingface.local_dir)
    path = snapshot_download(repo_id=repo_id, token=hf_token, local_dir=local_dir)
    return Path(path)


class InferRequest(BaseModel):
    date: str | None = None
    league_codes: list[str] | None = None


class InferResponse(BaseModel):
    predictions: list[dict[str, Any]]


class CustomPredictRequest(BaseModel):
    home_team: str
    away_team: str
    league_code: str


class CustomPredictResponse(BaseModel):
    home_team: str
    away_team: str
    predicted_outcome: str
    confidence: float
    probabilities: dict[str, float]
    league: str


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _PREDICTOR is not None}


@app.get("/teams")
def get_teams() -> dict[str, list[str]]:
    """Return available team names grouped by league code."""
    return _TEAMS_BY_LEAGUE


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    if _PREDICTOR is None or _CONFIG is None or _ENRICHED_DATA is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    target_date = req.date or datetime.now().strftime("%d/%m/%Y")
    fixtures = _fetch_fixtures(target_date, req.league_codes)
    if not fixtures:
        return InferResponse(predictions=[])

    enriched = _ENRICHED_DATA

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


@app.post("/predict-custom", response_model=CustomPredictResponse)
def predict_custom(req: CustomPredictRequest) -> CustomPredictResponse:
    """Predict a single matchup for any two teams, using league averages for unknowns."""
    if _PREDICTOR is None or _CONFIG is None or _ENRICHED_DATA is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    league_name = DIVISION_MAP.get(req.league_code, req.league_code)
    enriched = _ENRICHED_DATA

    fixture = {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "league": league_name,
        "division": req.league_code,
    }
    features = _build_features(fixture, enriched)
    feature_df = pd.DataFrame([features])

    try:
        preds = _PREDICTOR.predict(feature_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    if not preds:
        raise HTTPException(status_code=500, detail="Model returned no prediction")

    p = preds[0]
    d = p.to_dict()
    return CustomPredictResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        predicted_outcome=d.get("predicted_outcome", ""),
        confidence=float(d.get("confidence", 0.0)),
        probabilities=d.get("probabilities", {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}),
        league=league_name,
    )


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
        df = loader.load_all()
        if not df.empty:
            cleaner = DataCleaner()
            df = cleaner.clean(df)
        return df
    except Exception:
        return pd.DataFrame()


def _extract_teams_by_league(df: pd.DataFrame) -> dict[str, list[str]]:
    """Extract team names per league code from raw historical data."""
    if df.empty or "League" not in df.columns:
        return {}
    league_name_to_code = {v: k for k, v in DIVISION_MAP.items()}
    result: dict[str, list[str]] = {}
    for league_name, group in df.groupby("League"):
        code = league_name_to_code.get(str(league_name))
        if code:
            teams = sorted(set(group["HomeTeam"].unique()) | set(group["AwayTeam"].unique()))
            result[code] = teams
    return result


def _engineer_features(df: pd.DataFrame, config: SpaceConfig) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        fe = FeatureEngineer(config.features)
        df = fe.build_all_features(df)
        elo = FootballELO(config.features.elo)
        df = elo.compute_elo_features(df)
        return df
    except Exception:
        return pd.DataFrame()


def _build_features(
    fixture: dict[str, str], enriched: pd.DataFrame
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "HomeTeam": fixture["home_team"],
        "AwayTeam": fixture["away_team"],
    }
    if enriched.empty:
        return features

    league_avg = enriched.mean(numeric_only=True)

    home_rows = enriched[
        (enriched["HomeTeam"] == fixture["home_team"])
        | (enriched["AwayTeam"] == fixture["home_team"])
    ]
    away_rows = enriched[
        (enriched["HomeTeam"] == fixture["away_team"])
        | (enriched["AwayTeam"] == fixture["away_team"])
    ]

    home_last = home_rows.iloc[-1] if not home_rows.empty else league_avg
    away_last = away_rows.iloc[-1] if not away_rows.empty else league_avg
    home_was_home = home_last.get("HomeTeam", "") == fixture["home_team"]
    away_was_away = away_last.get("AwayTeam", "") == fixture["away_team"]

    numeric_cols = [
        c for c in enriched.columns
        if c not in ("HomeTeam", "AwayTeam", "FTR", "Date")
    ]
    for col in numeric_cols:
        if col.startswith("home_"):
            src = home_last if home_was_home else away_last
            opposite = "away_" + col[5:]
            val = src.get(col if home_was_home else opposite, 0)
        elif col.startswith("away_"):
            src = away_last if away_was_away else home_last
            opposite = "home_" + col[5:]
            val = src.get(col if away_was_away else opposite, 0)
        else:
            val = home_last.get(col, league_avg.get(col, 0))
        try:
            features[col] = float(val) if pd.notna(val) else 0.0
        except (TypeError, ValueError):
            features[col] = 0.0

    return features
