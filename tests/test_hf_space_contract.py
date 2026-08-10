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
    # The fixture feature row. Mirrored because a drifted copy here is exactly
    # the bug this module was created to remove: the Space and the scheduled
    # job predicting different outcomes for the same match.
    Path("src/models/fixture_features.py"),
    # Cross-league routing and the evidence counts behind a refusal.
    Path("src/models/cross_league.py"),
    # The cross-league model itself. The Space serves it on /predict-custom;
    # a drifted copy would answer with a different blend from the one the
    # offline sweep measured, under the same label.
    Path("src/models/european_predictor.py"),
    Path("src/models/outcome_model.py"),
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

    def test_space_carries_the_cross_league_model(self):
        """Stated separately from the byte comparison: an absent file makes
        that test fail with a FileNotFoundError, which reads as a broken test
        rather than as a Space that cannot answer /predict-custom."""
        assert (PROJECT_ROOT / "hf_space/src/models/european_predictor.py").exists()

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


class TestEuropeanPredictionConfigParity:
    """The two tuned numbers must be the same on both sides.

    ``dixon_coles_weight`` and ``elo_draw_rate`` were *measured*, over 1,301
    held-out European matches. A Space serving a different pair would produce
    untuned probabilities under a label claiming otherwise, and nothing
    downstream could tell — which is precisely the failure the ``model`` field
    exists to prevent, arriving through the config instead.
    """

    #: The values that came out of the sweep. Everything else in the block is
    #: a gate, not a fitted parameter.
    TUNED = ("dixon_coles_weight", "elo_draw_rate")

    @staticmethod
    def _space_block() -> dict:
        return _yaml("hf_space/config/config.yaml")["european_prediction"]

    @staticmethod
    def _main_block() -> dict:
        return _yaml("config/config.yaml")["european"]["prediction"]

    def test_space_loader_declares_the_config(self):
        module = _load_module(
            Path("hf_space/config/config_loader.py"), "space_loader_european"
        )
        assert hasattr(module, "EuropeanPredictionConfig")
        assert "european_prediction" in module.SpaceConfig.model_fields

    def test_space_config_declares_the_block(self):
        assert "european_prediction" in _yaml("hf_space/config/config.yaml")

    def test_tuned_values_match(self):
        space, main = self._space_block(), self._main_block()
        for key in self.TUNED:
            assert space[key] == main[key], f"{key} has drifted from the tuned value"

    def test_the_refusal_gate_matches(self):
        assert (
            self._space_block()["min_matches_per_team"]
            == self._main_block()["min_matches_per_team"]
        )

    def test_the_space_does_not_carry_the_corpus_keys(self):
        """It has no corpus builder, no providers and no alias store. Keys it
        cannot act on would only invite a divergence nobody notices."""
        block = self._space_block()
        for absent in ("providers", "competitions", "seasons", "cache_path"):
            assert absent not in block


class TestBothDeploymentsShareTheFixtureBuilder:
    """Neither side may grow its own copy of the fixture feature row again.

    They each had one. The Space's understood only ``home_``/``away_``
    prefixes and copied every ``diff_*``, ``elo_*`` and ``h2h_*`` from the home
    team's *previous* match, so for Estrela vs Sp Lisbon on 2026-08-08 it fed
    the model Sp Braga's rating as the home side's and predicted a 44% home win
    where the scheduled job's row gave a 74% away win. Byte-identical mirrors
    (above) only help if both call sites actually use the module.
    """

    @staticmethod
    def _source(path: str) -> str:
        return (PROJECT_ROOT / path).read_text()

    def test_the_scheduled_job_uses_the_shared_builder(self):
        assert "FixtureFeatureBuilder" in self._source("scripts/run_inference.py")

    def test_the_space_uses_the_shared_builder(self):
        assert "FixtureFeatureBuilder" in self._source("hf_space/app.py")

    def test_the_space_no_longer_walks_the_corpus_columns_itself(self):
        """The signature of the old implementation: iterating the dataframe's
        columns instead of the model's feature names."""
        assert "numeric_cols" not in self._source("hf_space/app.py")

    def test_the_scheduled_job_no_longer_branches_on_feature_names(self):
        source = self._source("scripts/run_inference.py")
        assert 'elif feat == "elo_diff"' not in source
