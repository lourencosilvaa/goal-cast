from config.config_loader import AnalysisConfig
from src.analysis.divergence import DivergenceAnalyzer


class TestDivergenceAnalyzer:

    def _make_config(self) -> AnalysisConfig:
        return AnalysisConfig(
            value_threshold=0.05,
            min_edge=0.03,
            blend_weights={"ml_model": 0.50, "bookmaker_avg": 0.30, "best_odds": 0.20},
        )

    def test_divergence_features_has_required_keys(self):
        analyzer = DivergenceAnalyzer(self._make_config())
        ml = {"home": 0.55, "draw": 0.25, "away": 0.20}
        bk = {"home": 0.48, "draw": 0.27, "away": 0.25}
        features = analyzer.compute_divergence_features(ml, bk)

        assert "kl_div_ml_bk" in features
        assert "max_divergence" in features
        assert "sources_agree" in features
        assert "blended_prob_H" in features

    def test_identical_probs_zero_kl_divergence(self):
        analyzer = DivergenceAnalyzer(self._make_config())
        probs = {"home": 0.45, "draw": 0.30, "away": 0.25}
        features = analyzer.compute_divergence_features(probs, probs)

        assert abs(features["kl_div_ml_bk"]) < 0.001
        assert features["max_divergence"] < 0.001
        assert features["sources_agree"] == 1.0

    def test_divergent_probs_positive_kl(self):
        analyzer = DivergenceAnalyzer(self._make_config())
        ml = {"home": 0.60, "draw": 0.20, "away": 0.20}
        bk = {"home": 0.40, "draw": 0.30, "away": 0.30}
        features = analyzer.compute_divergence_features(ml, bk)

        assert features["kl_div_ml_bk"] > 0
        assert features["max_divergence"] > 0.1

    def test_sources_disagree_on_favorite(self):
        analyzer = DivergenceAnalyzer(self._make_config())
        ml = {"home": 0.50, "draw": 0.25, "away": 0.25}
        bk = {"home": 0.25, "draw": 0.25, "away": 0.50}
        features = analyzer.compute_divergence_features(ml, bk)

        assert features["sources_agree"] == 0.0

    def test_blended_probabilities_sum_close_to_one(self):
        analyzer = DivergenceAnalyzer(self._make_config())
        ml = {"home": 0.50, "draw": 0.30, "away": 0.20}
        bk = {"home": 0.45, "draw": 0.30, "away": 0.25}
        features = analyzer.compute_divergence_features(ml, bk)

        blended_total = (
            features["blended_prob_H"]
            + features["blended_prob_D"]
            + features["blended_prob_A"]
        )
        assert abs(blended_total - 1.0) < 0.05

    def test_multi_source_divergence(self):
        analyzer = DivergenceAnalyzer(self._make_config())
        ml = {"home": 0.55, "draw": 0.25, "away": 0.20}
        sources = [
            {"home": 0.50, "draw": 0.27, "away": 0.23},
            {"home": 0.48, "draw": 0.28, "away": 0.24},
            {"home": 0.52, "draw": 0.26, "away": 0.22},
        ]
        features = analyzer.compute_multi_source_divergence(ml, sources)

        assert "bk_std_H" in features
        assert "bk_consensus" in features
        assert features["bk_consensus"] > 0

    def test_multi_source_empty_list(self):
        analyzer = DivergenceAnalyzer(self._make_config())
        ml = {"home": 0.50, "draw": 0.25, "away": 0.25}
        features = analyzer.compute_multi_source_divergence(ml, [])

        assert features == {}
