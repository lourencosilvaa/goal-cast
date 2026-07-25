"""End-to-end: train a tiny international model then predict via the predictor.

Exercises the real ``InternationalMatchPredictor.__init__``/``prepare`` path
and confirms the goals-only pipeline trains and serves a coherent prediction.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config_loader import load_config
from src.models.data_cleaner import DataCleaner
from src.models.international.data_loader import InternationalDataLoader
from src.models.international.elo import InternationalELO
from src.models.international.feature_engineer import InternationalFeatureEngineer
from src.models.international.predictor import InternationalMatchPredictor
from src.models.trainer import ModelTrainer


def _synthetic_results(path: Path) -> None:
    teams = ["Portugal", "Spain", "France", "Brazil", "Argentina", "Germany"]
    rng = np.random.default_rng(11)
    base = pd.Timestamp("2005-01-01")
    rows = []
    for i in range(300):
        h, a = rng.choice(teams, size=2, replace=False)
        hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        rows.append(
            {
                "date": (base + pd.Timedelta(days=10 * i)).strftime("%Y-%m-%d"),
                "home_team": h,
                "away_team": a,
                "home_score": hg,
                "away_score": ag,
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": bool(i % 4 == 0),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_train_and_predict_end_to_end(tmp_path: Path):
    dataset = tmp_path / "results.csv"
    _synthetic_results(dataset)
    models_dir = tmp_path / "models"

    config = load_config()
    config.international.dataset_path = str(dataset)
    config.international.models_dir = str(models_dir)
    config.international.min_date = "2000-01-01"
    # Disable Poisson to keep the integration test fast and self-contained.
    if config.model.poisson:
        config.model.poisson.enabled = False

    # --- Train (mirrors scripts/train_international_model.py core) ---
    raw = InternationalDataLoader(config.international).load_all()
    clean = DataCleaner().clean(raw)
    featured = InternationalFeatureEngineer(config.features).build_all_features(clean)
    elo = InternationalELO(
        config.features.elo,
        neutral_home_advantage_factor=config.international.neutral_home_advantage_factor,
    )
    featured = elo.compute_elo_features(featured)

    trainer = ModelTrainer(config.model)
    X, y, _ = trainer.prepare_data(featured)
    trainer.build_ensemble(X, y, dates=featured["Date"])
    trainer.save_model(str(models_dir))

    assert (models_dir / "ensemble_model.joblib").exists()

    # --- Predict via the real predictor path ---
    predictor = InternationalMatchPredictor(config).prepare()
    prediction = predictor.predict("Portugal", "Spain", neutral=True)

    probs = [
        prediction.home_win_prob,
        prediction.draw_prob,
        prediction.away_win_prob,
    ]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-6
