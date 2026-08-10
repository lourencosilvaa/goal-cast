"""HuggingFace Space — Football Prediction FastAPI service.

Loads the trained model once at startup, then serves on-demand predictions.
Accepts POST /infer with {date, league_codes} and returns predictions.

It also serves the statistics endpoints (/match-insights, /team-insights):
the Space is the only deployed component holding the full match history in
memory, so empirical team and head-to-head numbers are computed here and
proxied by the backend.
"""

import csv
import io
import time as _time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from config.config_loader import SpaceConfig, load_config
from src.analysis.team_insights import (
    FixtureQuery,
    TeamInsightsCalculator,
    TeamQuery,
)
from src.models.cross_competition import (
    CrossCompetitionCorpus,
    CrossCompetitionEloBuilder,
)
from src.models.cross_league import is_cross_league, match_counts
from src.models.data_cleaner import DataCleaner
from src.models.data_loader import FootballDataLoader
from src.models.elo import FootballELO
from src.models.european_predictor import EuropeanMatchPredictor, PredictionRefused
from src.models.feature_engineer import FeatureEngineer
from src.models.fixture_features import FixtureFeatureBuilder, FixtureTeams
from src.models.predictor import MatchPredictor

_PREDICTOR: MatchPredictor | None = None
_CONFIG: SpaceConfig | None = None
_ENRICHED_DATA: pd.DataFrame | None = None
_TEAMS_BY_LEAGUE: dict[str, list[str]] = {}
_INSIGHTS: TeamInsightsCalculator | None = None

#: The ELO instance left behind by the cross-league walk, and how much history
#: each club has. Both are startup state rather than per-request work: the
#: ratings are the *only* place the cross-league calibration survives (the walk
#: returns domestic rows, so a European result moves a rating without ever
#: appearing in the output), and recounting appearances on every request would
#: be a full pass over the corpus to answer a question that cannot change.
_ELO: FootballELO | None = None
_MATCH_COUNTS: dict[str, int] = {}

_FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"
_FIXTURES_CACHE = Path("/tmp/fixtures.csv")
_FIXTURES_CACHE_AGE = 60 * 60  # 1 hour

#: Division code → league name, populated from ``data.leagues`` at startup.
#: Holding a literal copy here meant every league added to the config had to be
#: mirrored by hand, and a missed edit silently mislabelled fixtures. Reads
#: before startup fall back to the raw division code, which is also the right
#: display for a division the config does not cover.
DIVISION_MAP: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _PREDICTOR, _CONFIG, _ENRICHED_DATA, _TEAMS_BY_LEAGUE, _INSIGHTS
    global _ELO, _MATCH_COUNTS
    _CONFIG = load_config()
    DIVISION_MAP.update(_CONFIG.data.leagues)
    model_dir = _download_model(_CONFIG)
    _PREDICTOR = MatchPredictor(model_dir)
    # Pre-compute enriched features once at startup
    historical_df = _load_historical_data(_CONFIG)
    # Extract teams per league from raw data before enrichment
    _TEAMS_BY_LEAGUE = _extract_teams_by_league(historical_df)
    # The insight calculator reads raw results (goals, dates, shots), so it is
    # built from the cleaned frame before feature engineering rewrites it.
    _INSIGHTS = _build_insights(historical_df, _CONFIG, _PREDICTOR)
    _ENRICHED_DATA, _ELO = _engineer_features(historical_df, _CONFIG)
    _MATCH_COUNTS = match_counts(_ENRICHED_DATA)
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
    #: The away side's league, when it differs from the home side's. Absent
    #: means "the same one" — which is what every caller sending a single code
    #: has always meant, and must keep meaning.
    away_league_code: str | None = None


class CustomPredictResponse(BaseModel):
    home_team: str
    away_team: str
    predicted_outcome: str
    confidence: float
    probabilities: dict[str, float]
    league: str
    #: Set only on a cross-league answer; the home league alone is ambiguous
    #: once the two sides can differ.
    away_league: str | None = None
    #: What actually produced the numbers. ``None`` is the domestic ensemble —
    #: naming it on every response would leave the cross-league label meaning
    #: nothing by contrast.
    model: str | None = None


class MatchInsightsRequest(BaseModel):
    home_team: str
    away_team: str
    league_code: str


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
    """Predict a single matchup for any two teams.

    Which model answers is decided by the *pairing*, not by a competition
    badge. The ensemble is trained only on rows where both sides share a
    league, so it has never seen a cross-league fixture and carries no feature
    that would tell it one had arrived — a Liga Portugal club against a Premier
    League club is that fixture whether or not a European badge is attached.
    """
    if _PREDICTOR is None or _CONFIG is None or _ENRICHED_DATA is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    away_league_code = req.away_league_code or req.league_code
    if is_cross_league(req.league_code, away_league_code):
        return _predict_cross_league(req, away_league_code)

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
        probabilities=d.get(
            "probabilities", {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
        ),
        league=league_name,
    )


def _predict_cross_league(
    req: CustomPredictRequest, away_league_code: str
) -> CustomPredictResponse:
    """Price a fixture whose two sides come from different leagues.

    Dixon-Coles and ELO, because the corpus calibration made those two
    comparable across league pools and left the ensemble untouched. A club with
    no history is refused with the reason stated (422) rather than answered
    with a number invented from a default rating — the same refusal the name
    resolver makes with spellings it has not been taught.
    """
    if _CONFIG is None or not _CONFIG.european_prediction.enabled:
        raise HTTPException(
            status_code=503, detail="Cross-league prediction is disabled"
        )
    # Dixon-Coles answers the *history* question even at weight zero: it is the
    # only component that knows which clubs exist at all, so without it the
    # refusal gate cannot run and every answer would be a guess.
    if _PREDICTOR is None or _PREDICTOR.poisson is None or _ELO is None:
        raise HTTPException(status_code=503, detail="Cross-league model not loaded")

    predictor = EuropeanMatchPredictor(
        dixon_coles=_PREDICTOR.poisson,
        elo=_ELO,
        config=_CONFIG.european_prediction,
        match_counts=_MATCH_COUNTS,
    )
    try:
        prediction = predictor.predict(req.home_team, req.away_team)
    except PredictionRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    payload = prediction.to_dict()
    probabilities: dict[str, float] = payload["probabilities"]  # type: ignore[assignment]
    return CustomPredictResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        predicted_outcome=str(payload["predicted_outcome"]),
        confidence=float(payload["confidence"]),  # type: ignore[arg-type]
        probabilities=probabilities,
        league=DIVISION_MAP.get(req.league_code, req.league_code),
        away_league=DIVISION_MAP.get(away_league_code, away_league_code),
        model=prediction.model,
    )


@app.post("/match-insights")
def match_insights(req: MatchInsightsRequest) -> dict[str, Any]:
    """Head-to-head history plus both team profiles for a face-off."""
    calculator = _require_insights()
    query = FixtureQuery(
        league_code=req.league_code,
        home_team=req.home_team,
        away_team=req.away_team,
    )
    for team in (req.home_team, req.away_team):
        _require_known_team(calculator, req.league_code, team)
    return calculator.match_insights(query).to_dict()


@app.get("/team-insights")
def team_insights(
    team: str = Query(...), league_code: str = Query(...)
) -> dict[str, Any]:
    """Full statistical profile of a single team."""
    calculator = _require_insights()
    _require_known_team(calculator, league_code, team)
    return calculator.team_insights(
        TeamQuery(league_code=league_code, team=team)
    ).to_dict()


def _require_insights() -> TeamInsightsCalculator:
    if _INSIGHTS is None:
        raise HTTPException(status_code=503, detail="Historical data not loaded")
    return _INSIGHTS


def _require_known_team(
    calculator: TeamInsightsCalculator, league_code: str, team: str
) -> None:
    if not calculator.knows_team(TeamQuery(league_code=league_code, team=team)):
        raise HTTPException(
            status_code=404,
            detail=f"No history for '{team}' in league '{league_code}'",
        )


def _build_insights(
    df: pd.DataFrame, config: SpaceConfig, predictor: MatchPredictor | None
) -> TeamInsightsCalculator:
    """Wire the calculator to the raw history and the fitted score model."""
    return TeamInsightsCalculator(
        matches=df,
        config=config.insights,
        leagues=config.data.leagues,
        market_model=predictor.poisson if predictor is not None else None,
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
            teams = sorted(
                set(group["HomeTeam"].unique()) | set(group["AwayTeam"].unique())
            )
            result[code] = teams
    return result


#: Cross-league history uploaded alongside the per-league Parquet datasets.
#: Already translated to canonical team keys, so the Space needs no registry
#: or alias source of its own.
EUROPEAN_DATASET = "european.parquet"


def _load_european_corpus(config: SpaceConfig) -> pd.DataFrame:
    """European results the model was calibrated on, or an empty frame.

    Absent on an older snapshot, which must keep working — it just means the
    ELO ratings are not comparable across leagues, exactly as before.
    """
    path = (
        Path(config.huggingface.local_dir)
        / config.huggingface.dataset_subfolder
        / EUROPEAN_DATASET
    )
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _engineer_features(
    df: pd.DataFrame, config: SpaceConfig
) -> tuple[pd.DataFrame, FootballELO]:
    """The model's feature matrix, **and the ELO instance that produced it**.

    Returning the instance is not a convenience. ``build`` yields domestic rows
    only, so every European result updates a rating without appearing in the
    output — recomputing ELO from that frame would look equivalent and silently
    revert each rating to its uncalibrated value, which is the entire thing the
    corpus work exists to prevent. The instance is where the calibration lives.
    """
    elo = FootballELO(config.features.elo)
    if df.empty:
        return df, elo
    try:
        fe = FeatureEngineer(config.features)
        df = fe.build_all_features(df)
        # The same builder and corpus training used. A fixture's elo_home is
        # read from that team's last historical row, so reproducing it means
        # replaying the identical chronological walk — serving uncalibrated
        # ratings to a model trained on calibrated ones is worse than never
        # calibrating, because the features stop meaning what it learned.
        european = _load_european_corpus(config)
        built = CrossCompetitionEloBuilder(elo).build(
            CrossCompetitionCorpus(domestic=df, supplementary=european)
        )
        return built, elo
    except Exception:
        return pd.DataFrame(), elo


def _build_features(fixture: dict[str, str], enriched: pd.DataFrame) -> dict[str, Any]:
    """The model's feature row for an unplayed fixture.

    Delegates to the shared
    :class:`~src.models.fixture_features.FixtureFeatureBuilder`, which
    ``scripts/run_inference.py`` also uses. This function used to carry its own
    implementation, and it only understood ``home_``/``away_`` prefixes: every
    ``diff_*``, ``elo_*`` and ``h2h_*`` was copied from the home team's
    *previous* match, so the Space and the scheduled job predicted different
    outcomes for the same fixture. Whatever this row needs to contain is now
    decided in exactly one place.

    It also asks for the predictor's own ``feature_names`` rather than every
    numeric column of the corpus: the model consumes that list, and building
    the rest was work with no reader.
    """
    if enriched.empty or _PREDICTOR is None:
        return {"HomeTeam": fixture["home_team"], "AwayTeam": fixture["away_team"]}

    return FixtureFeatureBuilder(enriched).build(
        FixtureTeams(home=fixture["home_team"], away=fixture["away_team"]),
        _PREDICTOR.feature_names,
    )
