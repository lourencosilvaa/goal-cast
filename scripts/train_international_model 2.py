"""
Train the national-teams (international) ML ensemble.

Uses the fixed Kaggle dataset (results.csv), goals-only features and a
neutral-venue-aware ELO. Artifacts are written to a dedicated directory
(``international.models_dir``) so the club-league model is never touched.

Usage:
    uv run python scripts/train_international_model.py
    uv run python scripts/train_international_model.py --config config/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd

from config.config_loader import load_config
from src.models.data_cleaner import DataCleaner
from src.models.international.data_loader import InternationalDataLoader
from src.models.international.elo import InternationalELO
from src.models.international.feature_engineer import InternationalFeatureEngineer
from src.models.poisson.dixon_coles import DixonColesModel
from src.models.time_weighting import TimeDecayWeighter
from src.models.trainer import ModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train national-teams prediction model"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output", default=None, help="Override output directory for the model"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    intl = config.international
    output_dir = args.output or intl.models_dir

    print("\n=== Loading International Data ===")
    loader = InternationalDataLoader(intl)
    raw_data = loader.load_all()
    if raw_data.empty:
        print("ERROR: No international data loaded. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(raw_data)} matches from {intl.dataset_path}")

    print("\n=== Cleaning Data ===")
    clean_data = DataCleaner().clean(raw_data)
    print(f"After cleaning: {len(clean_data)} matches")

    print("\n=== Feature Engineering (goals-only) ===")
    engineer = InternationalFeatureEngineer(config.features)
    featured_data = engineer.build_all_features(clean_data)
    print(f"Matches with features: {len(featured_data)}")

    print("\n=== Computing Neutral-Aware ELO ===")
    elo = InternationalELO(
        config.features.elo,
        neutral_home_advantage_factor=intl.neutral_home_advantage_factor,
    )
    featured_data = elo.compute_elo_features(featured_data)
    print("Top 5 national teams by ELO:")
    for team, rating in elo.get_top_teams(5):
        print(f"  {team:25s} {rating:.0f}")

    print("\n=== Training Models ===")
    weighter = (
        TimeDecayWeighter(config.model.time_decay)
        if config.model.time_decay
        else None
    )
    trainer = ModelTrainer(config.model, weighter=weighter)
    X, y, feature_names = trainer.prepare_data(featured_data)
    dates = featured_data["Date"]
    print(f"Features: {len(feature_names)}  Samples: {len(X)}")

    print("\n=== Cross-Validation ===")
    cv_results = trainer.cross_validate(X, y, dates=dates)

    print("\n=== Building Ensemble ===")
    ensemble_results = trainer.build_ensemble(X, y, dates=dates)

    poisson_cfg = config.model.poisson
    poisson: DixonColesModel | None = None
    if poisson_cfg and poisson_cfg.enabled:
        print("\n=== Training Dixon-Coles Poisson Model ===")
        poisson = DixonColesModel(poisson_cfg).fit(clean_data)
        print(
            f"Fitted {len(poisson.attack)} teams; "
            f"home advantage: {poisson.home_advantage:.3f}, rho: {poisson.rho:.3f}"
        )

    latest_date: pd.Timestamp = pd.to_datetime(featured_data["Date"]).max()
    last_match_date_str = (
        latest_date.strftime("%Y-%m-%d") if not pd.isna(latest_date) else None
    )
    trainer.save_model(output_dir, last_match_date=last_match_date_str)

    if poisson is not None:
        poisson_path = Path(output_dir) / "poisson_model.joblib"
        joblib.dump(poisson, poisson_path)
        print(f"Poisson model saved to {poisson_path}")

    results_path = Path(output_dir) / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(
            {
                "cross_validation": cv_results,
                "ensemble": ensemble_results,
                "features_count": len(feature_names),
                "samples_count": len(X),
                "feature_names": feature_names,
                "last_match_date": last_match_date_str,
                "track": "international",
            },
            f,
            indent=2,
        )
    print(f"\nTraining results saved to {results_path}")


if __name__ == "__main__":
    main()
