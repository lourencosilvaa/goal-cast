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
