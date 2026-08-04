from pathlib import Path

from config.config_loader import (
    InternationalConfig,
    InternationalFlashScoreConfig,
    load_config,
)


class TestInternationalConfig:
    def _config_path(self) -> Path:
        return Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"

    def test_root_config_exposes_international(self):
        config = load_config(self._config_path())
        assert isinstance(config.international, InternationalConfig)

    def test_dataset_path_present(self):
        config = load_config(self._config_path())
        assert config.international.dataset_path.endswith("results.csv")

    def test_models_dir_is_dedicated(self):
        config = load_config(self._config_path())
        assert "international" in config.international.models_dir

    def test_flashscore_leagues_configured(self):
        config = load_config(self._config_path())
        assert isinstance(
            config.international.flashscore, InternationalFlashScoreConfig
        )
        assert len(config.international.flashscore.leagues) > 0

    def test_neutral_factor_default(self):
        cfg = InternationalConfig(dataset_path="x/results.csv")
        assert 0.0 <= cfg.neutral_home_advantage_factor <= 1.0

    def test_tournaments_defaults_to_empty(self):
        cfg = InternationalConfig(dataset_path="x/results.csv")
        assert cfg.tournaments == []
