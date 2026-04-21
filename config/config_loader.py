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


class ScrapersConfig(BaseModel):
    request_timeout: int
    rate_limit_seconds: float
    user_agent: str
    betclic: ScraperSiteConfig
    betano: ScraperSiteConfig
    solverde: ScraperSiteConfig


class AnalysisConfig(BaseModel):
    value_threshold: float
    min_edge: float
    blend_weights: dict[str, float]


class OutputConfig(BaseModel):
    reports_dir: str
    models_dir: str
    plots_dir: str
    exports_dir: str = "output/exports"


class FootballDataOrgConfig(BaseModel):
    base_url: str = "https://api.football-data.org/v4"
    api_key: str = ""
    request_timeout: int = 10
    competitions: dict[str, str] = {}


class OddsAPIConfig(BaseModel):
    api_key: str = ""
    regions: str = "eu"
    markets: str = "h2h"


class RetrainCheckConfig(BaseModel):
    enabled: bool = True


class EvaluationConfig(BaseModel):
    storage_dir: str = "output/evaluation"


class Config(BaseModel):
    app: AppConfig
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    scrapers: ScrapersConfig
    analysis: AnalysisConfig
    output: OutputConfig
    football_data_org: FootballDataOrgConfig = FootballDataOrgConfig()
    odds_api: OddsAPIConfig = OddsAPIConfig()
    retrain_check: RetrainCheckConfig = RetrainCheckConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


def load_config(path: str | Path = "config/config.yaml") -> Config:
    """Load and validate configuration from a YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        data = yaml.safe_load(f)

    return Config(**data)
