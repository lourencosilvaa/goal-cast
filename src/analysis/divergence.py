import numpy as np

from config.config_loader import AnalysisConfig


class DivergenceAnalyzer:
    """
    Analyzes divergences between ML model probabilities
    and bookmaker odds. Divergences indicate potential value.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config

    def compute_divergence_features(
        self,
        ml_probs: dict[str, float],
        bookmaker_probs: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute features based on divergences between
        ML model and bookmaker probability sources.
        """
        features: dict[str, float] = {}
        epsilon = 1e-6

        for prefix, probs in [("ml", ml_probs), ("bk", bookmaker_probs)]:
            features[f"{prefix}_prob_H"] = probs.get("home", 0)
            features[f"{prefix}_prob_D"] = probs.get("draw", 0)
            features[f"{prefix}_prob_A"] = probs.get("away", 0)

        # KL-divergence: ML vs Bookmaker
        kl_div = 0.0
        for key in ["home", "draw", "away"]:
            p = max(ml_probs.get(key, epsilon), epsilon)
            q = max(bookmaker_probs.get(key, epsilon), epsilon)
            kl_div += p * np.log(p / q)
        features["kl_div_ml_bk"] = float(kl_div)

        # Absolute divergences per outcome
        for key, label in [("home", "H"), ("draw", "D"), ("away", "A")]:
            ml = ml_probs.get(key, 0)
            bk = bookmaker_probs.get(key, 0)
            features[f"divergence_{label}"] = ml - bk
            features[f"abs_divergence_{label}"] = abs(ml - bk)

        features["max_divergence"] = max(
            features["abs_divergence_H"],
            features["abs_divergence_D"],
            features["abs_divergence_A"],
        )

        # Do sources agree on the favorite?
        ml_favorite = max(ml_probs, key=lambda k: ml_probs[k])
        bk_favorite = max(bookmaker_probs, key=lambda k: bookmaker_probs[k])
        features["sources_agree"] = float(ml_favorite == bk_favorite)

        # Blended probabilities
        ml_w = self.config.blend_weights.get("ml_model", 0.5)
        bk_w = self.config.blend_weights.get("bookmaker_avg", 0.5)
        total_w = ml_w + bk_w
        for key, label in [("home", "H"), ("draw", "D"), ("away", "A")]:
            ml = ml_probs.get(key, 0)
            bk = bookmaker_probs.get(key, 0)
            features[f"blended_prob_{label}"] = (ml_w * ml + bk_w * bk) / total_w

        return features

    def compute_multi_source_divergence(
        self,
        ml_probs: dict[str, float],
        bookmaker_sources: list[dict[str, float]],
    ) -> dict[str, float]:
        """
        Compute divergences between ML and multiple bookmaker sources.
        """
        features: dict[str, float] = {}

        if not bookmaker_sources:
            return features

        # Average bookmaker probabilities
        avg_bk: dict[str, float] = {}
        for key in ["home", "draw", "away"]:
            values = [s.get(key, 0) for s in bookmaker_sources]
            avg_bk[key] = sum(values) / len(values) if values else 0

        features.update(self.compute_divergence_features(ml_probs, avg_bk))

        # Cross-bookmaker agreement (std deviation)
        for key, label in [("home", "H"), ("draw", "D"), ("away", "A")]:
            values = [s.get(key, 0) for s in bookmaker_sources]
            features[f"bk_std_{label}"] = float(np.std(values)) if values else 0

        features["bk_consensus"] = 1.0 - float(
            np.mean([features.get(f"bk_std_{l}", 0) for l in ["H", "D", "A"]])
        )

        return features
