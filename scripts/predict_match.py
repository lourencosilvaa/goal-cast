"""
Predict a specific match outcome using the trained model.

Usage:
    uv run python scripts/predict_match.py --home "Arsenal" --away "Brighton" --league "Premier League"
    uv run python scripts/predict_match.py --home "Benfica" --away "Porto" --league "Liga Portugal"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from src.models.predictor import MatchPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict match outcome")
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument("--league", default="", help="League name")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model-dir", default=None, help="Path to trained model directory"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_dir = args.model_dir or config.output.models_dir

    if not Path(model_dir).exists():
        print(
            f"ERROR: Model directory '{model_dir}' not found. "
            f"Train the model first with: uv run python scripts/train_model.py"
        )
        sys.exit(1)

    print(f"\n=== Predicting: {args.home} vs {args.away} ===")
    predictor = MatchPredictor(model_dir)

    # For a real prediction, we'd need to compute features for this match.
    # This script demonstrates the prediction interface.
    # The Copilot agent would orchestrate the full pipeline.
    print(
        f"\nModel loaded with {len(predictor.feature_names)} features."
    )
    print(
        "Note: For full predictions, use the find_value_bets.py script "
        "which combines data loading, feature engineering, and prediction."
    )

    result = {
        "match": f"{args.home} vs {args.away}",
        "league": args.league,
        "model_dir": model_dir,
        "features_count": len(predictor.feature_names),
        "status": "model_ready",
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
