from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class AppConfig(BaseModel):
    name: str
    version: str


class DataConfig(BaseModel):
    """Two league sets, deliberately different sizes.

    ``leagues`` is the **corpus**: every division the loader downloads, the
    trainer learns from and the ELO walk spans. ``served_leagues`` is the
    **product**: the subset a user can see and select.

    Keeping the corpus wider is not an oversight. The secondary divisions are
    roughly half the training rows, and a global ELO pool means they are the
    only reason a promoted side arrives carrying a real rating instead of the
    default — so they stay trained on long after they stop being offered.
    """

    base_url: str
    seasons: list[str]
    leagues: dict[str, str]
    #: Codes offered by the API and the UI. Must be a non-empty subset of
    #: ``leagues``. Required on purpose: defaulting to "all" would silently
    #: re-serve every division the moment the key went missing (§7.4).
    served_leagues: list[str]
    columns_to_keep: list[str]

    @field_validator("served_leagues")
    @classmethod
    def _served_must_be_a_configured_subset(
        cls, value: list[str], info: ValidationInfo
    ) -> list[str]:
        if not value:
            raise ValueError("served_leagues must not be empty")
        # `leagues` is declared first, so it is already validated and present
        # in `info.data` — unless it failed, in which case its own error is
        # the one worth reporting and this check has nothing to compare to.
        leagues = info.data.get("leagues")
        if leagues is None:
            return value
        unknown = [code for code in value if code not in leagues]
        if unknown:
            raise ValueError(
                f"served_leagues contains codes absent from leagues: "
                f"{', '.join(sorted(unknown))}"
            )
        return value


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
    #: ``False`` means never refit automatically — ``--force`` only.
    enabled: bool = True
    #: Matches that must accumulate since the model's last training date before
    #: a refit is worth its cost. Time-decay weighting makes a single round of
    #: matches negligible against a multi-season corpus, so refitting on one
    #: buys a redeploy and a service restart for no measurable gain. 0 refits
    #: on any new data. Scale it with the configured leagues: one weekly round
    #: is roughly ``len(data.leagues) * 9`` matches.
    min_new_matches: int = 0


class InferenceConfig(BaseModel):
    enabled: bool = True
    space_url: str = ""


class EvaluationConfig(BaseModel):
    storage_dir: str = "output/evaluation"


class InsightsConfig(BaseModel):
    """Windows and limits behind the team / head-to-head statistics.

    Every horizon the insight calculator uses is declared here rather than
    baked into the code, so the depth of "recent form" can be retuned from
    YAML without touching a module.
    """

    #: Matches counted as "recent" for form, rates and per-match averages.
    recent_matches: int = 10
    #: Past meetings surfaced in the head-to-head list (counts use all of them).
    h2h_matches: int = 10
    #: Length of the W/D/L streak shown in the UI.
    form_sequence_length: int = 5
    #: Most likely scorelines kept in the goal-market block.
    max_scorelines: int = 5


class TeamAliasConfig(BaseModel):
    """Canonical-name resolution for scraped team names.

    ``seed_path`` points at the versioned alias file whose entries are
    human-validated by code review; admin-approved aliases live in Supabase.
    The suggestion settings only drive *advisory* proposals shown to an admin —
    a close match is never applied automatically.
    """

    seed_path: str = "config/team_aliases.yaml"
    #: How many candidate canonical names to propose for an unresolved name.
    #: This — not the cutoff — is what keeps the admin's list short.
    suggestion_count: int = 5
    #: Minimum difflib similarity (0..1) for a candidate to be proposed.
    #: Deliberately below difflib's 0.6 default: measured against real pairs,
    #: the abbreviations this feature exists for score 0.40-0.53
    #: ("Sporting CP"->"Sp Lisbon" 0.40, "Wolverhampton"->"Wolves" 0.53), so
    #: 0.6 would hide the very candidates an admin needs to see.
    suggestion_cutoff: float = 0.4


class TeamsConfig(BaseModel):
    """Static team-name registry used as the offline fallback for /teams.

    The file ships inside ``src/backend/`` so the deployed backend image
    carries it without needing ``datasets/`` or pandas.
    """

    registry_path: str = "src/backend/data/teams_registry.json"
    #: Every team in the cached corpus, not just this season's. Alias
    #: resolution matches against this: a club relegated out of a tracked
    #: division keeps years of history the model still knows, and must stay
    #: matchable to it. Regenerate with ``--all-seasons``.
    historical_registry_path: str = "src/backend/data/teams_registry_historical.json"
    aliases: TeamAliasConfig = TeamAliasConfig()


class HuggingFaceConfig(BaseModel):
    repo_id: str = ""
    hf_token: str = ""
    local_dir: str = "/tmp/hf_models"
    model_filename: str = "ensemble_model.joblib"
    dataset_subfolder: str = "datasets"
    #: Space repo (``owner/name``) restarted after a retrain upload. The Space
    #: reads its match history once at boot, so a new dataset snapshot only
    #: reaches users once it reboots. Distinct from ``repo_id``, which is the
    #: model repo holding the artefacts and the Parquet datasets.
    space_repo_id: str = ""


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


class EuropeanTuningConfig(BaseModel):
    """Settings for the offline sweep that measures the blend parameters.

    Defined before :class:`EuropeanPredictionConfig` for the same reason
    :class:`ProviderConfig` is: a forward reference leaves the parent model
    not-fully-defined until ``model_rebuild()`` runs, which breaks anything
    loading this file as a standalone module.

    Used only by ``scripts/tune_european_weight.py``. Nothing at prediction
    time reads these, and the sweep never writes back — the measured values
    are adopted by editing ``dixon_coles_weight`` and ``elo_draw_rate`` by
    hand, exactly as the domestic blend weight is.
    """

    #: Candidate Dixon-Coles shares (0 = ELO only, 1 = Dixon-Coles only).
    blend_weight_grid: list[float] = Field(
        default_factory=lambda: [round(i / 20, 2) for i in range(21)]
    )
    #: Candidate draw shares for the ELO leg. Swept alongside the weight
    #: because the two interact: an ELO leg carving the draw out wrongly
    #: scores worse than it should and drags the weight toward Dixon-Coles.
    draw_rate_grid: list[float] = Field(
        default_factory=lambda: [round(0.16 + i * 0.02, 2) for i in range(9)]
    )
    #: How many recent seasons to hold out, one rolling-origin fold each.
    #: Only about half the corpus has both teams in a tracked league, so a
    #: single season is too thin to separate neighbouring weights.
    holdout_seasons: int = Field(default=5, ge=1)
    #: Month a European season starts. A December tie belongs to the season
    #: that began that summer, not to the calendar year.
    season_start_month: int = Field(default=7, ge=1, le=12)
    #: Folds with fewer evaluable matches than this are dropped rather than
    #: pooled, since their contribution is noise.
    min_holdout_matches: int = Field(default=30, ge=0)
    #: Resamples used for the interval on the improvement over the incumbent.
    bootstrap_samples: int = Field(default=2000, ge=0)
    bootstrap_seed: int = 42
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)

    @field_validator("blend_weight_grid")
    @classmethod
    def _weights_are_shares(cls, grid: list[float]) -> list[float]:
        if not grid:
            raise ValueError("blend_weight_grid must not be empty")
        if any(not 0.0 <= w <= 1.0 for w in grid):
            raise ValueError("blend_weight_grid values must lie in [0, 1]")
        return grid

    @field_validator("draw_rate_grid")
    @classmethod
    def _draw_rates_leave_room_to_win(cls, grid: list[float]) -> list[float]:
        if not grid:
            raise ValueError("draw_rate_grid must not be empty")
        if any(not 0.0 <= r < 1.0 for r in grid):
            raise ValueError("draw_rate_grid values must lie in [0, 1)")
        return grid


class EuropeanPredictionConfig(BaseModel):
    """How cross-league fixtures are predicted.

    The ensemble is deliberately absent. It is trained only on domestic rows —
    European results carry no shot, foul or corner columns, so they never enter
    its training matrix — meaning it has never seen a fixture whose two teams
    come from different leagues, and has no feature that would tell it one had
    arrived. Dixon-Coles and ELO were both made comparable across leagues by
    the corpus calibration, so they are what predict here.
    """

    enabled: bool = True
    #: Share of the blend taken from Dixon-Coles; the rest comes from
    #: ELO-implied probabilities.
    #:
    #: Zero, and measured rather than assumed. The blend was swept over 1,301
    #: held-out European matches (docs §14) and the log-loss profile rises
    #: monotonically with this weight at every draw rate: Dixon-Coles never
    #: improves a cross-league 1X2. Its attack and defence parameters rest on
    #: 2,636 European matches against 63,724 domestic ones, so they still
    #: describe a club's own league more than they describe Europe.
    #:
    #: Kept configurable rather than removed: the corpus grows every season,
    #: and ``scripts/tune_european_weight.py`` is how this gets revisited.
    dixon_coles_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Below this many recorded matches a team's parameters are noise rather
    #: than knowledge, and the fixture is refused instead of guessed.
    min_matches_per_team: int = 10
    #: ELO produces an expected *score*, with draws folded into it, so a draw
    #: probability has to be carved back out to reach a 1X2 triple. This is
    #: that share, stated rather than derived: ELO does not model draws at all,
    #: and deriving one from the rating gap would invent precision the model
    #: does not have.
    #:
    #: With ``dixon_coles_weight`` at zero this is the model's only remaining
    #: free parameter, so it was measured in the same sweep rather than left
    #: at the hand-picked 0.25.
    elo_draw_rate: float = Field(default=0.22, ge=0.0, lt=1.0)
    #: Offline-sweep settings. Read by the tuning script, never at prediction
    #: time, so a config predating this section changes nothing.
    tuning: EuropeanTuningConfig = Field(default_factory=EuropeanTuningConfig)


class ProviderConfig(BaseModel):
    """One upcoming-fixture API.

    Defined before :class:`EuropeanConfig` rather than after it with a string
    annotation: a forward reference leaves the parent model not-fully-defined
    until ``model_rebuild()`` runs, which breaks anything loading this file as
    a standalone module — the HF Space contract test does exactly that.

    Only the *name* of the environment variable holding the key lives here,
    never the key itself, so the file stays committable.

    ``competitions`` maps this project's competition codes to whatever the
    provider calls them: a sport key for The Odds API
    (``soccer_uefa_champs_league``), a competition code for football-data.org
    (``CL``). A code absent from the map is simply not requested — which is how
    "this provider covers the Champions League but not the Europa League" is
    expressed without any code change.
    """

    enabled: bool = True
    base_url: str = ""
    #: Environment variable holding the API key.
    api_key_env: str = ""
    #: Our competition code → the provider's identifier for it.
    competitions: dict[str, str] = Field(default_factory=dict)
    #: Seconds before a request is abandoned.
    timeout: int = 15
    #: Retries for transport-level errors only — never for an HTTP status,
    #: which would burn quota to receive the same answer.
    retries: int = 2


class EuropeanConfig(BaseModel):
    """UEFA club-competition track configuration.

    These competitions have no football-data.co.uk feed. Their results come
    from the public-domain openfootball project, parsed into a goals-only
    corpus that feeds ELO and Dixon-Coles only — the rows carry no shot, foul
    or corner columns, so the ensemble would drop them anyway.

    That corpus is what links the otherwise near-closed domestic league pools,
    without which ratings from different leagues are not comparable.
    """

    enabled: bool = True
    #: Abort rather than proceed uncalibrated when the corpus or the approved
    #: aliases are missing. Off by default, because degrading quietly is right
    #: for inference — a missing cache must not take predictions down. It is
    #: wrong for a scheduled retrain, which would otherwise overwrite a
    #: calibrated model with an uncalibrated one and report success.
    required: bool = False
    repo_url: str = "https://github.com/openfootball/champions-league.git"
    #: Where the shallow clone lives. Gitignored — it is a cache, not source.
    checkout_path: str = "datasets/openfootball/champions-league"
    #: Competition code → openfootball file stem, for the main draws.
    competitions: dict[str, str] = {}
    #: Qualifying rounds, kept separate: they bring in teams from leagues that
    #: are not tracked at all, so whether they help is measured, not assumed.
    qualifier_competitions: dict[str, str] = {}
    #: Competition code → the name shown in the UI. Kept apart from
    #: ``competitions`` (which maps to openfootball file stems) so a display
    #: string never doubles as a lookup key.
    competition_names: dict[str, str] = {}
    #: Seasons to ingest, in football-data's ``YYZZ`` form. Upstream coverage
    #: is uneven (Europa League starts 2020-21, Conference 2021-22, qualifiers
    #: 2024-25); a configured season with no file is skipped.
    seasons: list[str] = Field(default_factory=list)
    cache_path: str = "datasets/european/results.csv"
    qualifiers_cache_path: str = "datasets/european/results_with_qualifiers.csv"
    #: openfootball country code → the leagues of that country in
    #: ``data.leagues``. This narrows name-matching candidates to one country,
    #: which is what makes matching across competitions safe: without it
    #: "AC Sparta Praha" is close enough to "Sparta Rotterdam" to be accepted
    #: by mistake. A country with no tracked league yields no candidates, so
    #: its teams stay honestly unresolved.
    country_leagues: dict[str, list[str]] = {}
    #: Scope approved European aliases are stored under. Shared across the
    #: competitions so a club approved once is recognised in all of them —
    #: a side plays the Champions League one year and the Europa League the
    #: next, and should not need approving twice.
    alias_scope: str = "EU"
    #: Canonical name → human-facing name, so the UI can show "Sporting Clube
    #: de Portugal" while every model stays keyed by "Sp Lisbon".
    display_names_path: str = "config/team_display_names.yaml"
    #: Upcoming-fixture sources, tried in ``provider_order``.
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    #: Which providers to try, and in what order. Order is a correctness
    #: property, not a preference: the chain returns the first non-empty
    #: answer, so the source with the best coverage must come first.
    provider_order: list[str] = Field(default_factory=list)
    #: How many days ahead to look for fixtures. The Odds API only lists
    #: matches bookmakers are already pricing, so a longer window costs
    #: nothing but also reveals nothing.
    lookahead_days: int = 14
    #: How cross-league fixtures are predicted, once discovered.
    prediction: EuropeanPredictionConfig = EuropeanPredictionConfig()


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
    insights: InsightsConfig = InsightsConfig()
    huggingface: HuggingFaceConfig = HuggingFaceConfig()
    inference: InferenceConfig = InferenceConfig()
    international: InternationalConfig = InternationalConfig(
        dataset_path="datasets/international/results.csv"
    )
    european: EuropeanConfig = EuropeanConfig()


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
    if space_repo_id := os.environ.get("HF_SPACE_REPO_ID"):
        hf["space_repo_id"] = space_repo_id

    inference = data.setdefault("inference", {})
    if space_url := os.environ.get("HF_SPACE_URL"):
        inference["space_url"] = space_url

    return Config(**data)
