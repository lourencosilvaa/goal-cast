from pathlib import Path

from config.config_loader import (
    Config,
    load_config,
)


class TestConfigLoader:

    def test_load_config_returns_config_object(self, config_path: Path):
        config = load_config(config_path)
        assert isinstance(config, Config)

    def test_config_app_name(self, config_path: Path):
        config = load_config(config_path)
        assert config.app.name == "football-prediction-agent"

    def test_config_data_has_leagues(self, config_path: Path):
        config = load_config(config_path)
        assert len(config.data.leagues) > 0
        assert "E0" in config.data.leagues

    def test_config_data_has_seasons(self, config_path: Path):
        config = load_config(config_path)
        assert len(config.data.seasons) > 0

    def test_config_features_rolling_window(self, config_path: Path):
        config = load_config(config_path)
        assert config.features.rolling_window == 5

    def test_config_elo_settings(self, config_path: Path):
        config = load_config(config_path)
        assert config.features.elo.k_factor == 32
        assert config.features.elo.home_advantage == 65
        assert config.features.elo.initial_rating == 1500.0

    def test_config_model_settings(self, config_path: Path):
        config = load_config(config_path)
        assert config.model.test_size == 0.2
        assert config.model.random_state == 42

    def test_config_analysis_thresholds(self, config_path: Path):
        config = load_config(config_path)
        assert config.analysis.value_threshold > 0
        assert config.analysis.min_edge > 0

    def test_config_missing_file_raises_error(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_config_output_paths(self, config_path: Path):
        config = load_config(config_path)
        assert config.output.reports_dir != ""
        assert config.output.models_dir != ""
        assert config.output.plots_dir != ""

    def test_space_repo_id_defaults_to_empty(self, config_path: Path, monkeypatch):
        monkeypatch.delenv("HF_SPACE_REPO_ID", raising=False)
        config = load_config(config_path)
        assert config.huggingface.space_repo_id == ""

    def test_space_repo_id_read_from_environment(self, config_path: Path, monkeypatch):
        monkeypatch.setenv("HF_SPACE_REPO_ID", "tester/goal-cast-space")
        config = load_config(config_path)
        assert config.huggingface.space_repo_id == "tester/goal-cast-space"

    def test_empty_space_repo_id_env_does_not_override(
        self, config_path: Path, monkeypatch
    ):
        """An exported-but-blank secret must not read as a configured value."""
        monkeypatch.setenv("HF_SPACE_REPO_ID", "")
        config = load_config(config_path)
        assert config.huggingface.space_repo_id == ""

    def test_min_new_matches_holds_refits_to_roughly_monthly(self, config_path: Path):
        """~190 matches land per weekly round across the configured leagues."""
        config = load_config(config_path)
        assert config.retrain_check.min_new_matches == 800


class TestLeagueAndSeasonCoverage:
    """Coverage is a contract: the Space and the trainer must agree on it."""

    #: football-data.co.uk main divisions carrying the full stat schema. EC is
    #: deliberately absent — it ships no shot/foul/corner columns, so every one
    #: of its rows is dropped by FeatureEngineer.build_match_features.
    EXPECTED_LEAGUES = {
        "E0", "E1", "E2", "E3",
        "SC0", "SC1", "SC2", "SC3",
        "D1", "D2", "I1", "I2", "SP1", "SP2", "F1", "F2",
        "N1", "B1", "P1", "T1", "G1",
    }

    def _raw(self, path: Path) -> dict:
        import yaml

        with path.open() as handle:
            data: dict = yaml.safe_load(handle)
        return data

    def test_all_full_schema_divisions_are_configured(self, config_path: Path):
        config = load_config(config_path)
        assert set(config.data.leagues) == self.EXPECTED_LEAGUES

    def test_national_league_is_excluded(self, config_path: Path):
        config = load_config(config_path)
        assert "EC" not in config.data.leagues

    def test_current_season_is_configured(self, config_path: Path):
        config = load_config(config_path)
        assert "2627" in config.data.seasons

    def test_space_config_matches_root_config(self, config_path: Path):
        """A divergence here silently trains and serves different corpora."""
        space_path = config_path.parent.parent / "hf_space" / "config" / "config.yaml"
        root = self._raw(config_path)["data"]
        space = self._raw(space_path)["data"]
        assert sorted(space["seasons"]) == sorted(root["seasons"])
        assert space["leagues"] == root["leagues"]


class TestEuropeanCompetitionsConfig:
    """The UEFA club track must be fully declared in YAML.

    Nothing about these competitions may be implied at runtime: the repository
    URL, the competition→file map, the seasons and both cache paths are all
    environment-facing values (§6.1), asserted explicitly here (§7.3).
    """

    EXPECTED_COMPETITIONS = {"CL", "EL", "UECL"}
    EXPECTED_QUALIFIERS = {"CLQ", "ELQ", "UECLQ"}

    def test_track_is_enabled(self, config_path: Path):
        assert load_config(config_path).european.enabled is True

    def test_repo_url_points_at_openfootball(self, config_path: Path):
        config = load_config(config_path)
        assert "openfootball" in config.european.repo_url

    def test_all_three_competitions_are_configured(self, config_path: Path):
        config = load_config(config_path)
        assert set(config.european.competitions) == self.EXPECTED_COMPETITIONS

    def test_qualifiers_are_configured_separately(self, config_path: Path):
        """Kept apart so including them stays a measured decision."""
        config = load_config(config_path)
        assert set(config.european.qualifier_competitions) == self.EXPECTED_QUALIFIERS

    def test_qualifiers_are_not_mixed_into_the_main_competitions(
        self, config_path: Path
    ):
        config = load_config(config_path)
        overlap = set(config.european.competitions) & set(
            config.european.qualifier_competitions
        )
        assert overlap == set()

    def test_seasons_match_the_domestic_corpus(self, config_path: Path):
        """Time-decay weighting only treats both corpora alike if the season
        span is identical."""
        config = load_config(config_path)
        assert sorted(config.european.seasons) == sorted(config.data.seasons)

    def test_both_cache_paths_are_configured_and_distinct(self, config_path: Path):
        config = load_config(config_path)
        assert config.european.cache_path != config.european.qualifiers_cache_path
        assert config.european.cache_path.endswith(".csv")

    def test_checkout_path_is_configured(self, config_path: Path):
        config = load_config(config_path)
        assert config.european.checkout_path


class TestServedLeagues:
    """The product's league surface, which is narrower than the corpus.

    ``data.leagues`` drives the loader and the ELO walk; ``data.served_leagues``
    is what a user can actually see and select. The two are deliberately
    different: the secondary divisions are half the training rows and the only
    reason a promoted side arrives carrying a real rating, so they stay in the
    corpus long after they stop being offered.
    """

    #: First tiers only. Secondary divisions are trained on, never served.
    EXPECTED_SERVED = {
        "E0", "SC0", "SP1", "D1", "I1", "F1", "N1", "B1", "P1", "T1", "G1",
    }

    #: Trained on, deliberately absent from the product.
    EXPECTED_UNSERVED = {
        "E1", "E2", "E3", "SC1", "SC2", "SC3", "SP2", "D2", "I2", "F2",
    }

    def test_served_leagues_are_the_first_tiers(self, config_path: Path):
        config = load_config(config_path)
        assert set(config.data.served_leagues) == self.EXPECTED_SERVED

    def test_secondary_divisions_are_not_served(self, config_path: Path):
        config = load_config(config_path)
        served = set(config.data.served_leagues)
        assert served.isdisjoint(self.EXPECTED_UNSERVED)

    def test_secondary_divisions_remain_in_the_training_corpus(
        self, config_path: Path
    ):
        """Narrowing the corpus would cost 51% of rows and reset promoted ELO."""
        config = load_config(config_path)
        assert self.EXPECTED_UNSERVED <= set(config.data.leagues)

    def test_served_is_a_subset_of_the_corpus(self, config_path: Path):
        config = load_config(config_path)
        assert set(config.data.served_leagues) <= set(config.data.leagues)


class TestServedLeaguesValidation:
    """A served code the loader will never fetch must fail at load time."""

    def _data(self, **overrides: object) -> dict:
        base: dict = {
            "base_url": "https://example.test/",
            "seasons": ["2526"],
            "leagues": {"E0": "Premier League", "E1": "Championship"},
            "served_leagues": ["E0"],
            "columns_to_keep": ["Date"],
        }
        base.update(overrides)
        return base

    def test_accepts_a_subset(self):
        from config.config_loader import DataConfig

        assert DataConfig(**self._data()).served_leagues == ["E0"]

    def test_rejects_a_code_absent_from_leagues(self):
        import pytest
        from pydantic import ValidationError

        from config.config_loader import DataConfig

        with pytest.raises(ValidationError, match="ZZ9"):
            DataConfig(**self._data(served_leagues=["E0", "ZZ9"]))

    def test_rejects_an_empty_list(self):
        import pytest
        from pydantic import ValidationError

        from config.config_loader import DataConfig

        with pytest.raises(ValidationError):
            DataConfig(**self._data(served_leagues=[]))

    def test_is_required(self):
        """No silent fallback to 'all leagues' (§7.4) — a missing key fails."""
        import pytest
        from pydantic import ValidationError

        from config.config_loader import DataConfig

        payload = self._data()
        del payload["served_leagues"]
        with pytest.raises(ValidationError):
            DataConfig(**payload)


class TestServedLeaguesEdges:
    """Boundaries found while wiring the served set through the consumers."""

    def _data(self, **overrides: object) -> dict:
        base: dict = {
            "base_url": "https://example.test/",
            "seasons": ["2526"],
            "leagues": {"E0": "Premier League", "E1": "Championship"},
            "served_leagues": ["E0"],
            "columns_to_keep": ["Date"],
        }
        base.update(overrides)
        return base

    def test_serving_the_whole_corpus_is_allowed(self):
        """Nothing forces the sets apart — the narrowing is a choice."""
        from config.config_loader import DataConfig

        config = DataConfig(**self._data(served_leagues=["E0", "E1"]))
        assert set(config.served_leagues) == set(config.leagues)

    def test_order_is_preserved(self):
        """Config order drives display order in the pickers."""
        from config.config_loader import DataConfig

        data = self._data(
            leagues={"E0": "a", "E1": "b", "SP1": "c"},
            served_leagues=["SP1", "E0"],
        )
        assert DataConfig(**data).served_leagues == ["SP1", "E0"]

    def test_duplicates_are_tolerated_not_fatal(self):
        """A repeated code is sloppy, not dangerous — consumers use a set."""
        from config.config_loader import DataConfig

        assert DataConfig(**self._data(served_leagues=["E0", "E0"])).served_leagues == [
            "E0",
            "E0",
        ]

    def test_unknown_codes_are_all_reported_at_once(self):
        import pytest
        from pydantic import ValidationError

        from config.config_loader import DataConfig

        with pytest.raises(ValidationError) as excinfo:
            DataConfig(**self._data(served_leagues=["AA1", "ZZ9"]))
        message = str(excinfo.value)
        assert "AA1" in message and "ZZ9" in message

    def test_space_config_is_not_required_to_carry_the_served_set(
        self, config_path: Path
    ):
        """The Space supplies the catalogue; the backend scopes it.

        Duplicating served_leagues there would create a second definition that
        could drift from this one, so its absence is deliberate.
        """
        import yaml

        space = config_path.parent.parent / "hf_space" / "config" / "config.yaml"
        raw = yaml.safe_load(space.read_text(encoding="utf-8"))
        assert "served_leagues" not in raw["data"]
