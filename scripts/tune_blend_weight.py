"""Offline sweep to pick the ensemble/Poisson blend weight empirically.

Fits the calibrated ensemble and the Dixon-Coles model on a chronological
train split, then evaluates every candidate blend weight (model.poisson.
blend_weight_grid) on the held-out tail by multiclass log-loss. Report-only:
review the table and set model.poisson.blend_weight by hand.

Usage:
    uv run python scripts/tune_blend_weight.py
    uv run python scripts/tune_blend_weight.py --local-data datasets/joined_data.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from src.models.blend_search import BlendWeightSweeper
from src.models.data_cleaner import DataCleaner
from src.models.data_loader import FootballDataLoader
from src.models.elo import FootballELO
from src.models.feature_engineer import FeatureEngineer
from src.models.poisson.dixon_coles import DixonColesModel
from src.models.time_weighting import TimeDecayWeighter
from src.models.trainer import ModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep the Poisson blend weight")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--local-data", default=None, help="Path to local CSV data file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    poisson_cfg = config.model.poisson
    if poisson_cfg is None or not poisson_cfg.enabled:
        print("model.poisson is disabled in config. Nothing to sweep.")
        sys.exit(0)

    print("\n=== Loading & preparing data ===")
    loader = FootballDataLoader(config.data)
    raw = (
        loader.load_from_csv(args.local_data) if args.local_data else loader.load_all()
    )
    if raw.empty:
        print("ERROR: No data loaded. Exiting.")
        sys.exit(1)

    clean = DataCleaner().clean(raw)
    featured = FeatureEngineer(config.features).build_all_features(clean)
    featured = FootballELO(config.features.elo).compute_elo_features(featured)

    weighter = (
        TimeDecayWeighter(config.model.time_decay) if config.model.time_decay else None
    )
    trainer = ModelTrainer(config.model, weighter=weighter)
    X, y, _ = trainer.prepare_data(featured)
    dates = featured["Date"]

    # Chronological split: fit on the head, evaluate the blend on the tail.
    split = int(len(X) * (1 - config.model.test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    boundary = dates.iloc[split]
    featured_test = featured.iloc[split:]

    print(f"Train: {len(X_train)}  Test: {len(X_test)}  (split at {boundary.date()})")

    # Calibrated ensemble fit on the training head only.
    print("\n=== Fitting calibrated ensemble (train split) ===")
    trainer.build_ensemble(X_train, y_train, dates=dates.iloc[:split])
    ensemble_proba = trainer.ensemble.predict_proba(trainer.scaler.transform(X_test))

    # Dixon-Coles fit on matches strictly before the test period.
    print("=== Fitting Dixon-Coles (train split) ===")
    poisson = DixonColesModel(poisson_cfg).fit(clean[clean["Date"] < boundary])

    # Poisson 1X2 per test fixture, columns ordered [away, draw, home] to
    # match the ensemble's class order (0=away, 1=draw, 2=home).
    poisson_rows = []
    for row in featured_test.itertuples():
        probs = poisson.predict_outcome(
            str(row.HomeTeam), str(row.AwayTeam)
        ).normalized()
        poisson_rows.append([probs.away_win, probs.draw, probs.home_win])
    poisson_proba = np.array(poisson_rows)

    print("\n=== Blend-weight sweep (held-out log-loss) ===")
    result = BlendWeightSweeper(poisson_cfg.blend_weight_grid).sweep(
        ensemble_proba, poisson_proba, y_test.to_numpy()
    )

    print(f"{'weight':>8}  {'log_loss':>10}")
    for w, ll in result.per_weight:
        marker = "  <- best" if w == result.best_weight else ""
        print(f"{w:>8.2f}  {ll:>10.4f}{marker}")
    print(
        f"\nBest blend_weight: {result.best_weight:.2f} "
        f"(log-loss {result.best_log_loss:.4f}). "
        f"Current config: {poisson_cfg.blend_weight:.2f}."
    )
    print("Report-only: update model.poisson.blend_weight in config.yaml to adopt.")


if __name__ == "__main__":
    main()
