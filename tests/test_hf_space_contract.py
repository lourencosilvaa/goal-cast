"""Deployment contract for the HuggingFace Space bundle.

``hf_space/`` is uploaded to the Space as a self-contained folder, so it
carries its own copy of the modules it needs (``hf_space/src/models/*`` are
copies of ``src/models/*``). Copies drift silently, and a drifted statistics
module means the deployed Space computes different numbers from the ones this
test suite verifies. These tests fail loudly instead.

Regenerate a stale mirror with:

    cp src/analysis/team_insights.py hf_space/src/analysis/team_insights.py
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIRRORED_MODULES = [
    Path("src/analysis/team_insights.py"),
    Path("src/models/elo.py"),
    Path("src/models/feature_engineer.py"),
    Path("src/models/predictor.py"),
    Path("src/models/data_cleaner.py"),
]


def _load_module(path: Path, name: str) -> ModuleType:
    """Import a file by path — the two config loaders share a module name."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _yaml(path: str) -> dict:
    with (PROJECT_ROOT / path).open() as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict)
    return data


class TestMirroredModules:

    def test_space_carries_the_insights_module(self):
        assert (PROJECT_ROOT / "hf_space/src/analysis/team_insights.py").exists()

    def test_space_analysis_package_is_importable(self):
        assert (PROJECT_ROOT / "hf_space/src/analysis/__init__.py").exists()

    def test_mirrors_are_byte_identical(self):
        drifted = [
            str(module)
            for module in MIRRORED_MODULES
            if (PROJECT_ROOT / module).read_bytes()
            != (PROJECT_ROOT / "hf_space" / module).read_bytes()
        ]
        assert not drifted, f"hf_space copies have drifted: {drifted}"


class TestInsightsConfigParity:
    """Both loaders must agree, or the Space reports over different horizons."""

    @staticmethod
    def _defaults(path: str) -> dict:
        module = _load_module(Path(path), f"loader_{path.replace('/', '_')}")
        return module.InsightsConfig().model_dump()

    def test_space_loader_declares_insights_config(self):
        module = _load_module(
            Path("hf_space/config/config_loader.py"), "space_loader"
        )
        assert hasattr(module, "InsightsConfig")
        assert "insights" in module.SpaceConfig.model_fields

    def test_defaults_match(self):
        assert self._defaults("config/config_loader.py") == self._defaults(
            "hf_space/config/config_loader.py"
        )


class TestShippedInsightsYaml:

    def test_main_config_declares_the_insights_block(self):
        assert "insights" in _yaml("config/config.yaml")

    def test_space_config_declares_the_insights_block(self):
        assert "insights" in _yaml("hf_space/config/config.yaml")

    def test_both_configs_agree(self):
        assert (
            _yaml("config/config.yaml")["insights"]
            == _yaml("hf_space/config/config.yaml")["insights"]
        )

    def test_horizons_are_positive(self):
        insights = _yaml("config/config.yaml")["insights"]
        assert all(value > 0 for value in insights.values())
