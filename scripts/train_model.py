"""
Train the ML ensemble model on historical football data.

Usage:
    uv run python scripts/train_model.py
    uv run python scripts/train_model.py --config config/config.yaml
    uv run python scripts/train_model.py --local-data datasets/joined_data.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib

from config.config_loader import load_config
from src.models.data_cleaner import DataCleaner
from src.models.data_loader import FootballDataLoader
from src.models.elo import FootballELO
from src.models.feature_engineer import FeatureEngineer
from src.models.model_cache_manager import ModelCacheManager
from src.models.poisson.dixon_coles import DixonColesModel
from src.models.time_weighting import TimeDecayWeighter
from src.models.trainer import ModelTrainer


def _emit_retrained(value: bool) -> None:
    """Signal to CI whether training actually ran.

    Writes ``retrained=true|false`` to the GitHub Actions step output so the
    workflow can skip the upload + redeploy steps when nothing was retrained
    (a no-new-data week). No-op outside GitHub Actions.
    """
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a") as fh:
            fh.write(f"retrained={'true' if value else 'false'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train football prediction model")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--local-data", default=None, help="Path to local CSV data file"
    )
    parser.add_argument(
        "--output", default=None, help="Output directory for model"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force retraining even if model is current"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output or config.output.models_dir

    # Load data
    print("\n=== Loading Data ===")
    loader = FootballDataLoader(config.data)
    if args.local_data:
        raw_data = loader.load_from_csv(args.local_data)
        print(f"Loaded {len(raw_data)} matches from {args.local_data}")
    else:
        raw_data = loader.load_all()

    if raw_data.empty:
        print("ERROR: No data loaded. Exiting.")
        sys.exit(1)

    # Check if retraining is needed
    import pandas as pd
    cache_manager = ModelCacheManager(models_dir=output_dir)
    latest_date: pd.Timestamp | None = None
    if "Date" in raw_data.columns:
        try:
            latest_date = pd.to_datetime(raw_data["Date"], dayfirst=True, errors="coerce").max()
        except Exception:
            pass

    if not args.force and not cache_manager.should_retrain(latest_data_date=latest_date):
        print(f"\nModel is up to date (last match date: {cache_manager.get_last_match_date().date()}). Skipping training.")
        print("Use --force to retrain anyway.")
        _emit_retrained(False)
        sys.exit(0)

    if latest_date:
        print(f"New data detected up to {latest_date.date()}. Retraining model...")

    # Clean data
    print("\n=== Cleaning Data ===")
    cleaner = DataCleaner()
    clean_data = cleaner.clean(raw_data)
    print(f"After cleaning: {len(clean_data)} matches")

    # Feature engineering
    print("\n=== Feature Engineering ===")
    engineer = FeatureEngineer(config.features)
    featured_data = engineer.build_all_features(clean_data)
    print(f"Matches with features: {len(featured_data)}")

    # ELO ratings
    print("\n=== Computing ELO Ratings ===")
    elo = FootballELO(config.features.elo)
    featured_data = elo.compute_elo_features(featured_data)
    print("Top 5 teams by ELO:")
    for team, rating in elo.get_top_teams(5):
        print(f"  {team:25s} {rating:.0f}")

    # Train models
    print("\n=== Training Models ===")
    weighter = (
        TimeDecayWeighter(config.model.time_decay)
        if config.model.time_decay
        else None
    )
    if weighter and weighter.enabled:
        half_life = config.model.time_decay.half_life_days
        print(f"Time-decay weighting enabled (half-life: {half_life:.0f} days)")
    trainer = ModelTrainer(config.model, weighter=weighter)
    X, y, feature_names = trainer.prepare_data(featured_data)
    dates = featured_data["Date"]

    print(f"\nFeatures: {len(feature_names)}")
    print(f"Samples: {len(X)}")

    # Cross-validation
    print("\n=== Cross-Validation ===")
    cv_results = trainer.cross_validate(X, y, dates=dates)

    # Build ensemble
    print("\n=== Building Ensemble ===")
    ensemble_results = trainer.build_ensemble(X, y, dates=dates)

    # Dixon-Coles Poisson score model (fit on cleaned goals data). Blended
    # into the ensemble 1X2 at inference and used for O/U 2.5 and BTTS.
    poisson_cfg = config.model.poisson
    if poisson_cfg and poisson_cfg.enabled:
        print("\n=== Training Dixon-Coles Poisson Model ===")
        poisson = DixonColesModel(poisson_cfg).fit(clean_data)
        print(f"Fitted {len(poisson.attack)} teams; "
              f"home advantage: {poisson.home_advantage:.3f}, rho: {poisson.rho:.3f}")

    # Save model
    last_match_date_str = latest_date.strftime("%Y-%m-%d") if latest_date and not pd.isna(latest_date) else None
    trainer.save_model(output_dir, last_match_date=last_match_date_str)

    if poisson_cfg and poisson_cfg.enabled:
        poisson_path = Path(output_dir) / "poisson_model.joblib"
        joblib.dump(poisson, poisson_path)
        print(f"Poisson model saved to {poisson_path}")

    # Collect the seasons that contributed data to the model
    seasons_trained: list[str] = []
    if "Season" in raw_data.columns:
        seasons_trained = sorted(raw_data["Season"].dropna().unique().tolist())

    # Save training results
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
                "seasons_trained": seasons_trained,
            },
            f,
            indent=2,
        )
    print(f"\nTraining results saved to {results_path}")
    if seasons_trained:
        print(f"Seasons trained: {', '.join(seasons_trained)}")

    _emit_retrained(True)


if __name__ == "__main__":
    main()
