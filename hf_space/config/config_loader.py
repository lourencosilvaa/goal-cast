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


class InsightsConfig(BaseModel):
    """Windows and limits behind the team / head-to-head statistics.

    Mirrors ``config.config_loader.InsightsConfig`` in the main project, which
    is where the defaults are documented.
    """

    #: Matches counted as "recent" for form, rates and per-match averages.
    recent_matches: int = 10
    #: Past meetings surfaced in the head-to-head list (counts use all of them).
    h2h_matches: int = 10
    #: Length of the W/D/L streak shown in the UI.
    form_sequence_length: int = 5
    #: Most likely scorelines kept in the goal-market block.
    max_scorelines: int = 5


class EuropeanPredictionConfig(BaseModel):
    """How a fixture whose two teams come from different leagues is priced.

    Deliberately a *slim* mirror of ``config.config_loader``'s
    ``EuropeanPredictionConfig``, and nothing else from ``EuropeanConfig``: the
    Space has no corpus builder, no fixture providers and no alias store, so
    carrying those keys here would only create somewhere for them to drift.

    ``dixon_coles_weight`` and ``elo_draw_rate`` were measured over 1,301
    held-out European matches, not chosen. ``tests/test_hf_space_contract.py``
    pins them to the main config for that reason — a Space serving a different
    pair would produce untuned probabilities under a tuned label.
    """

    enabled: bool = True
    #: Share of the blend taken from Dixon-Coles; the rest is ELO-implied.
    dixon_coles_weight: float = 0.0
    #: Below this many recorded matches a club's parameters are noise, and the
    #: fixture is refused rather than guessed.
    min_matches_per_team: int = 10
    #: ELO yields an expected *score* with draws folded in, so the draw share
    #: is stated rather than derived — ELO does not model draws at all.
    elo_draw_rate: float = 0.22


class SpaceConfig(BaseModel):
    data: DataConfig
    features: FeaturesConfig
    huggingface: HuggingFaceConfig = HuggingFaceConfig()
    insights: InsightsConfig = InsightsConfig()
    european_prediction: EuropeanPredictionConfig = EuropeanPredictionConfig()


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
