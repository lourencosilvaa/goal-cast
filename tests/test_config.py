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

    def test_config_scrapers_enabled(self, config_path: Path):
        config = load_config(config_path)
        assert config.scrapers.betclic.enabled is True
        assert config.scrapers.betano.enabled is True
        assert config.scrapers.solverde.enabled is True

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
