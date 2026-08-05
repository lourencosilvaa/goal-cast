from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str
    version: str


class DataConfig(BaseModel):
    base_url: str
    seasons: list[str]
    leagues: dict[str, str]
    columns_to_keep: list[str]


class EloConfig(BaseModel):
    k_factor: int
    home_advantage: int
    initial_rating: float


class XgProxyConfig(BaseModel):
    sot_conversion: float
    shot_conversion: float


class FatigueConfig(BaseModel):
    max_rest_days: int
    default_rest_days: int
    fatigue_threshold: int


class FeaturesConfig(BaseModel):
    rolling_window: int
    elo: EloConfig
    xg_proxy: XgProxyConfig
    fatigue: FatigueConfig


class LogisticRegressionConfig(BaseModel):
    max_iter: int
    C: float


class RandomForestConfig(BaseModel):
    n_estimators: int
    max_depth: int
    min_samples_leaf: int


class XGBoostConfig(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float


class EnsembleConfig(BaseModel):
    voting: str
    weights: list[int]


class TimeDecayConfig(BaseModel):
    """Exponential time-decay weighting for training samples."""

    enabled: bool
    half_life_days: float


class CalibrationConfig(BaseModel):
    """Probability calibration of the fitted ensemble.

    Calibration is fit on a held-out chronological slice (never the
    training rows) so the corrected probabilities are leakage-safe.
    """

    enabled: bool = False
    method: str = "sigmoid"  # "sigmoid" (Platt) or "isotonic"
    calibration_fraction: float = 0.15


class XGBSearchConfig(BaseModel):
    """Bounded, leakage-safe XGBoost log-loss hyperparameter search."""

    enabled: bool = True
    n_iter: int = 20
    n_splits: int = 5
    # int | float union so integer params (n_estimators, max_depth) stay
    # integers while continuous params (learning_rate, subsample) stay floats.
    param_grid: dict[str, list[int | float]] = {}


class PoissonConfig(BaseModel):
    """Dixon-Coles Poisson score model.

    Models goals directly to produce calibrated scorelines and the O/U 2.5
    and BTTS markets. Its 1X2 distribution is blended into the ensemble's
    with weight ``blend_weight`` (0 = ensemble only, 1 = Poisson only).
    ``blend_weight_grid`` is the candidate grid the offline sweep searches.
    """

    enabled: bool = False
    max_goals: int = 10
    half_life_days: float = 540
    blend_weight: float = 0.4
    blend_weight_grid: list[float] = Field(
        default_factory=lambda: [round(i / 10, 1) for i in range(11)]
    )


class ModelConfig(BaseModel):
    test_size: float
    random_state: int
    logistic_regression: LogisticRegressionConfig
    random_forest: RandomForestConfig
    xgboost: XGBoostConfig
    ensemble: EnsembleConfig
    time_decay: TimeDecayConfig | None = None
    calibration: CalibrationConfig | None = None
    xgb_search: XGBSearchConfig | None = None
    poisson: PoissonConfig | None = None


class FlashScoreConfig(BaseModel):
    base_url: str
    api_url: str
    token_url: str
    enabled: bool = True
    http_enabled: bool = True
    playwright_fallback_enabled: bool = True
    leagues: dict[str, str] = {}


class ScrapersConfig(BaseModel):
    request_timeout: int
    rate_limit_seconds: float
    user_agent: str
    flashscore: FlashScoreConfig = FlashScoreConfig(
        base_url="https://www.flashscore.com",
        api_url="https://d.flashscore.com/x/feed",
        token_url="https://www.flashscore.com",
    )


class AnalysisConfig(BaseModel):
    value_threshold: float
    min_edge: float
    blend_weights: dict[str, float]


class OutputConfig(BaseModel):
    reports_dir: str
    models_dir: str
    plots_dir: str
    exports_dir: str = "output/exports"


class OddsAPIConfig(BaseModel):
    api_key: str = ""
    regions: str = "eu"
    markets: str = "h2h"


class RetrainCheckConfig(BaseModel):
    enabled: bool = True


class InferenceConfig(BaseModel):
    enabled: bool = True
    space_url: str = ""


class EvaluationConfig(BaseModel):
    storage_dir: str = "output/evaluation"


class TeamsConfig(BaseModel):
    """Static team-name registry used as the offline fallback for /teams.

    The file ships inside ``src/backend/`` so the deployed backend image
    carries it without needing ``datasets/`` or pandas.
    """

    registry_path: str = "src/backend/data/teams_registry.json"


class HuggingFaceConfig(BaseModel):
    repo_id: str = ""
    hf_token: str = ""
    local_dir: str = "/tmp/hf_models"
    model_filename: str = "ensemble_model.joblib"
    dataset_subfolder: str = "datasets"


class SpaceConfig(BaseModel):
    huggingface: HuggingFaceConfig
    inference: InferenceConfig


class InternationalFlashScoreConfig(BaseModel):
    """FlashScore competition slug map for national-team fixtures.

    Keys are internal tournament codes and values are FlashScore URL slugs
    (e.g. ``world/world-cup``) resolved by the shared FlashScore scraper.
    """

    leagues: dict[str, str] = {}


class InternationalConfig(BaseModel):
    """National-teams (international) prediction track configuration.

    Drives a parallel, goals-only pipeline trained from a fixed Kaggle
    dataset. All environment-dependent values live here (never hardcoded):
    dataset path, training filters, neutral-venue handling, artifact
    directory and the FlashScore competition slug map.
    """

    enabled: bool = True
    dataset_path: str
    min_date: str = "1990-01-01"
    models_dir: str = "output/models/international"
    # Scales ELO/Poisson home advantage on neutral venues: 0.0 removes it
    # entirely (true neutral), 1.0 keeps the full home advantage.
    neutral_home_advantage_factor: float = 0.0
    # Empty list means "include every tournament in the dataset".
    tournaments: list[str] = Field(default_factory=list)
    flashscore: InternationalFlashScoreConfig = InternationalFlashScoreConfig()


class Config(BaseModel):
    app: AppConfig
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    scrapers: ScrapersConfig
    analysis: AnalysisConfig
    output: OutputConfig
    odds_api: OddsAPIConfig = OddsAPIConfig()
    retrain_check: RetrainCheckConfig = RetrainCheckConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    teams: TeamsConfig = TeamsConfig()
    huggingface: HuggingFaceConfig = HuggingFaceConfig()
    inference: InferenceConfig = InferenceConfig()
    international: InternationalConfig = InternationalConfig(
        dataset_path="datasets/international/results.csv"
    )


def load_config(path: str | Path = "config/config.yaml") -> Config:
    """Load and validate configuration from a YAML file."""
    import os

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        data = yaml.safe_load(f)

    hf = data.setdefault("huggingface", {})
    if repo_id := os.environ.get("HF_REPO_ID"):
        hf["repo_id"] = repo_id
    if hf_token := os.environ.get("HF_TOKEN"):
        hf["hf_token"] = hf_token
    if local_dir := os.environ.get("HF_LOCAL_DIR"):
        hf["local_dir"] = local_dir

    inference = data.setdefault("inference", {})
    if space_url := os.environ.get("HF_SPACE_URL"):
        inference["space_url"] = space_url

    return Config(**data)
