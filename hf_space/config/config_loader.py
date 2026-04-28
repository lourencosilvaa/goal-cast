import os
from pathlib import Path

import yaml
from pydantic import BaseModel


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


class HuggingFaceConfig(BaseModel):
    repo_id: str = ""
    hf_token: str = ""
    local_dir: str = "/tmp/hf_models"
    model_filename: str = "ensemble_model.joblib"
    dataset_subfolder: str = "datasets"


class SpaceConfig(BaseModel):
    data: DataConfig
    features: FeaturesConfig
    huggingface: HuggingFaceConfig = HuggingFaceConfig()


def load_config(path: str | Path = "config/config.yaml") -> SpaceConfig:
    config_path = Path(path)
    with config_path.open() as f:
        data = yaml.safe_load(f)

    hf = data.setdefault("huggingface", {})
    if repo_id := os.environ.get("HF_REPO_ID"):
        hf["repo_id"] = repo_id
    if hf_token := os.environ.get("HF_TOKEN"):
        hf["hf_token"] = hf_token
    if local_dir := os.environ.get("HF_LOCAL_DIR"):
        hf["local_dir"] = local_dir

    return SpaceConfig(**data)
