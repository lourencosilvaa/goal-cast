"""Offline, bounded XGBoost log-loss hyperparameter search.

Runs a leakage-safe (TimeSeriesSplit) randomized search over the XGBoost
parameter grid defined in ``model.xgb_search`` and reports the best
parameters and cross-validated log-loss. It is intentionally report-only:
review the output, then update ``model.xgboost`` in config.yaml by hand.

Usage:
    uv run python scripts/tune_xgboost.py
    uv run python scripts/tune_xgboost.py --config config/config.yaml
    uv run python scripts/tune_xgboost.py --local-data datasets/joined_data.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from src.models.data_cleaner import DataCleaner
from src.models.data_loader import FootballDataLoader
from src.models.elo import FootballELO
from src.models.feature_engineer import FeatureEngineer
from src.models.hyperparameter_search import XGBoostLogLossSearcher
from src.models.trainer import ModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune XGBoost via log-loss search")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--local-data", default=None, help="Path to local CSV data file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config.model.xgb_search is None or not config.model.xgb_search.enabled:
        print("xgb_search is disabled in config. Nothing to do.")
        sys.exit(0)

    print("\n=== Loading Data ===")
    loader = FootballDataLoader(config.data)
    raw_data = (
        loader.load_from_csv(args.local_data)
        if args.local_data
        else loader.load_all()
    )
    if raw_data.empty:
        print("ERROR: No data loaded. Exiting.")
        sys.exit(1)

    print("\n=== Cleaning & Feature Engineering ===")
    clean_data = DataCleaner().clean(raw_data)
    featured = FeatureEngineer(config.features).build_all_features(clean_data)
    featured = FootballELO(config.features.elo).compute_elo_features(featured)

    trainer = ModelTrainer(config.model)
    X, y, feature_names = trainer.prepare_data(featured)
    print(f"Features: {len(feature_names)}  Samples: {len(X)}")

    print("\n=== Bounded XGBoost Log-Loss Search (TimeSeriesSplit) ===")
    searcher = XGBoostLogLossSearcher(
        config.model.xgb_search, random_state=config.model.random_state
    )
    result = searcher.search(X, y)

    print(f"\nCandidates evaluated: {result.n_candidates}")
    print(f"Best CV log-loss:     {result.best_log_loss:.4f}")
    print("Best params:")
    for key, value in sorted(result.best_params.items()):
        print(f"  {key}: {value}")
    print(
        "\nReport-only: update model.xgboost in config.yaml by hand "
        "if you want to adopt these."
    )


if __name__ == "__main__":
    main()
