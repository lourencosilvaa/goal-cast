"""
Predict a national-team fixture with the international model.

Usage:
    uv run python scripts/predict_international_match.py --home Portugal --away Spain
    uv run python scripts/predict_international_match.py --home Brazil --away Argentina --neutral
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from src.models.international.predictor import (
    InternationalMatchPredictor,
    _models_available,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a national-team match")
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument(
        "--neutral", action="store_true", help="Match played on a neutral venue"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not _models_available(config.international.models_dir):
        print(
            "ERROR: International model not found. Train it first:\n"
            "  uv run python scripts/train_international_model.py"
        )
        sys.exit(1)

    predictor = InternationalMatchPredictor(config).prepare()
    prediction = predictor.predict(args.home, args.away, neutral=args.neutral)

    venue = "neutral venue" if args.neutral else f"{args.home} at home"
    print(f"\n=== {args.home} vs {args.away} ({venue}) ===")
    print(f"  Home win: {prediction.home_win_prob:.1%}")
    print(f"  Draw:     {prediction.draw_prob:.1%}")
    print(f"  Away win: {prediction.away_win_prob:.1%}")
    print(f"  Predicted: {prediction.predicted_outcome} "
          f"(confidence {prediction.confidence:.1%})")


if __name__ == "__main__":
    main()
