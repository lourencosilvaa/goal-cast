from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


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
    """FlashScore scraping, which is browser-rendered DOM reading and nothing else.

    ``api_url``, ``token_url`` and ``http_enabled`` used to live here for an
    HTTP feed client that could not authenticate — FlashScore moved the
    ``feed_sign`` token into a runtime JS object — so it raised on every call
    and the scraper fell straight through to Playwright. Configuration for a
    path that cannot run is worse than no configuration: it suggests a choice
    that does not exist.
    """

    base_url: str
    enabled: bool = True
    playwright_fallback_enabled: bool = True
    leagues: dict[str, str] = {}


class ScrapersConfig(BaseModel):
    request_timeout: int
    rate_limit_seconds: float
    user_agent: str
    flashscore: FlashScoreConfig = FlashScoreConfig(
        base_url="https://www.flashscore.com",
    )


class AnalysisConfig(BaseModel):
    value_threshold: float
    min_edge: float
    blend_weights: dict[str, float]


class OutputConfig(BaseModel):
    reports_dir: str
    models_dir: str
    plots_dir: str


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


class AiProviderConfig(BaseModel):
    """One match-analysis back-end.

    ``key_service`` is the row the user's encrypted key is stored under in
    ``user_api_keys`` — naming it here is what stops a provider reading
    somebody else's credential, which is exactly what happened while the
    endpoint fetched the Gemini key regardless of which back-end was meant.
    """

    #: Service name in the key store ("gemini", "nvidia").
    key_service: str
    #: Used when the caller names no model.
    default_model: str
    #: OpenAI-compatible base URL. Empty for providers reached through an SDK
    #: rather than raw HTTP, which is how Gemini is called.
    base_url: str = ""
    timeout: int = 60


class AiConfig(BaseModel):
    """Which back-ends can write a match analysis, and which one by default."""

    #: Used when a request names no provider. A configured value rather than a
    #: constant so switching the house default needs no deploy of new code.
    default_provider: str = "gemini"
    gemini: AiProviderConfig = AiProviderConfig(
        key_service="gemini", default_model="gemini-2.5-flash"
    )
    nvidia: AiProviderConfig = AiProviderConfig(
        key_service="nvidia",
        default_model="meta/llama-3.3-70b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
    )

    @field_validator("default_provider")
    @classmethod
    def _default_must_be_one_we_have(cls, value: str) -> str:
        known = ("gemini", "nvidia")
        if value not in known:
            raise ValueError(
                f"ai.default_provider must be one of {', '.join(known)}, got {value!r}"
            )
        return value


class ResultsProviderConfig(BaseModel):
    """One results source reached over HTTP.

    Deliberately a separate model from :class:`ProviderConfig` even though the
    fields rhyme: that one describes an *upcoming-fixture* API and is validated
    against a fixture chain. Sharing it would tie the two tracks together, and
    the results track already needs fields the fixture one does not (a live
    endpoint's competitions are the served leagues, not the UEFA codes).

    Only the *name* of the environment variable holding the key appears here,
    never the key, so the file stays committable.
    """

    enabled: bool = True
    base_url: str = ""
    #: Environment variable holding the API key.
    api_key_env: str = ""
    #: Our league code → the provider's identifier for it.
    competitions: dict[str, str] = Field(default_factory=dict)
    #: Seconds before a request is abandoned.
    timeout: int = 15
    #: Retries for transport-level errors only — never for an HTTP status.
    retries: int = 2


class LocalCorpusConfig(BaseModel):
    """The football-data.co.uk CSVs read as the first history source.

    ``search_dirs`` are consulted before anything is downloaded, which is what
    makes history free in development: the training cache already holds every
    season. The deployed service has no such cache, so it downloads into
    ``results.cache_dir`` on first use.
    """

    enabled: bool = True
    base_url: str = "https://www.football-data.co.uk/mmz4281"
    #: Directories searched for ``{season}_{league}.csv`` before downloading.
    search_dirs: list[str] = Field(default_factory=lambda: ["datasets/cache"])
    #: League codes that have a football-data.co.uk feed. A code absent from
    #: this list is never requested — the UEFA competitions have no such feed,
    #: and asking for one would 404 on every call.
    leagues: list[str] = Field(default_factory=list)
    timeout: int = 30


class ResultsLiveConfig(BaseModel):
    """The two clocks behind on-demand polling.

    ``poll_interval_seconds`` is the TTL: inside it, a cached snapshot is
    served and no request is made, which is what keeps a page left open from
    consuming the provider's quota. ``stale_after_seconds`` is the honesty
    threshold: past it, a served snapshot is flagged ``stale`` rather than
    presented as current.
    """

    poll_interval_seconds: int = Field(default=60, gt=0)
    stale_after_seconds: int = Field(default=300, gt=0)

    @field_validator("stale_after_seconds")
    @classmethod
    def _stale_must_not_precede_refresh(cls, value: int, info: ValidationInfo) -> int:
        interval = info.data.get("poll_interval_seconds")
        if interval is not None and value < interval:
            raise ValueError(
                "stale_after_seconds must not be shorter than "
                "poll_interval_seconds: a snapshot cannot go stale before the "
                "cache would have refreshed it"
            )
        return value


class ResultsServiceConfig(BaseModel):
    """Settings read by the dedicated results service process itself."""

    #: Environment variable holding the service-to-service API key. The
    #: service refuses to start when it is unset — see
    #: :mod:`src.results_service.auth`.
    api_key_env: str = "RESULTS_SERVICE_API_KEY"


class ResultsConfig(BaseModel):
    """History and live results — read by the dedicated service, not the app.

    The two provider orders are separate because the two jobs are: history is
    answered by a local corpus almost always, live never can be. Order is a
    correctness property in both, since the chain returns the first non-empty
    answer.

    ``enabled`` defaults to *false* here while the shipped ``config.yaml`` sets
    it true. That is deliberate: a config file with no ``results:`` section has
    not configured this track, and the honest reading of that is "off", not
    "on with providers I had to invent". The validations below only apply when
    it is switched on, so turning it on is what forces it to be coherent.
    """

    enabled: bool = False
    history_provider_order: list[str] = Field(
        default_factory=lambda: ["local_corpus", "football_data"]
    )
    live_provider_order: list[str] = Field(default_factory=lambda: ["football_data"])
    live: ResultsLiveConfig = ResultsLiveConfig()
    #: Where downloaded CSVs and live snapshots are persisted.
    cache_dir: str = "datasets/cache/results"
    service: ResultsServiceConfig = ResultsServiceConfig()
    local_corpus: LocalCorpusConfig = LocalCorpusConfig()
    providers: dict[str, ResultsProviderConfig] = Field(default_factory=dict)

    #: History source that is not an HTTP provider, so it has no ``providers``
    #: entry to be validated against.
    NON_HTTP_SOURCES: ClassVar[tuple[str, ...]] = ("local_corpus",)

    @model_validator(mode="after")
    def _orders_are_coherent_when_enabled(self) -> "ResultsConfig":
        if not self.enabled:
            return self
        for field in ("history_provider_order", "live_provider_order"):
            order: list[str] = getattr(self, field)
            if not order:
                raise ValueError(f"{field} must not be empty when results.enabled")
            unknown = [
                name
                for name in order
                if name not in self.providers and name not in self.NON_HTTP_SOURCES
            ]
            if unknown:
                raise ValueError(
                    f"{field} names providers absent from results.providers: "
                    f"{', '.join(sorted(unknown))}"
                )
        return self


class ResultsGatewayConfig(BaseModel):
    """How the main app reaches the results service.

    The URL is an environment variable name rather than a value because it
    differs per environment (an internal Render hostname in production,
    localhost in development), and the key must never be committed.
    """

    enabled: bool = True
    #: Environment variable holding the service's base URL.
    base_url_env: str = "RESULTS_SERVICE_URL"
    #: Environment variable holding the shared service key.
    api_key_env: str = "RESULTS_SERVICE_API_KEY"
    #: Shorter than the service's own provider timeouts on purpose: the app
    #: would rather report "results service unreachable" than hold a user's
    #: request open for as long as the slowest upstream provider might take.
    timeout: int = 10


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
    ai: AiConfig = AiConfig()
    #: Read by the dedicated results service only.
    results: ResultsConfig = ResultsConfig()
    #: Read by the main app only — how it reaches that service.
    results_gateway: ResultsGatewayConfig = ResultsGatewayConfig()


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
