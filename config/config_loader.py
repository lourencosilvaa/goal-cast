from pathlib import Path

import yaml
from pydantic import BaseModel


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


class ModelConfig(BaseModel):
    test_size: float
    random_state: int
    logistic_regression: LogisticRegressionConfig
    random_forest: RandomForestConfig
    xgboost: XGBoostConfig
    ensemble: EnsembleConfig


class ScraperSiteConfig(BaseModel):
    base_url: str
    enabled: bool


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
    betclic: ScraperSiteConfig
    betano: ScraperSiteConfig
    solverde: ScraperSiteConfig
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


class HuggingFaceConfig(BaseModel):
    repo_id: str = ""
    hf_token: str = ""
    local_dir: str = "/tmp/hf_models"
    model_filename: str = "ensemble_model.joblib"
    dataset_subfolder: str = "datasets"


class SpaceConfig(BaseModel):
    huggingface: HuggingFaceConfig
    inference: InferenceConfig

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
    huggingface: HuggingFaceConfig = HuggingFaceConfig()
    inference: InferenceConfig = InferenceConfig()


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
