"""Offline sweep to pick the European blend parameters empirically.

``european.prediction.dixon_coles_weight`` was set to 0.6 by analogy with the
domestic blend and ``elo_draw_rate`` to 0.25 as a rough observed rate. Neither
was measured. This measures both on held-out European matches, one
rolling-origin fold per recent season.

Report-only, exactly like ``scripts/tune_blend_weight.py``: review the tables
and edit the two values by hand if the result warrants it. It deliberately
does not adopt its own finding — on roughly 900 evaluable matches the winning
cell is often not distinguishable from what is already configured.

Usage:
    uv run python scripts/tune_european_weight.py
    uv run python scripts/tune_european_weight.py --max-folds 2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from src.corpus.european_corpus import load_european_corpus
from src.models.cross_competition import CrossCompetitionCorpus
from src.models.data_cleaner import DataCleaner
from src.models.data_loader import FootballDataLoader
from src.models.elo import FootballELO
from src.models.european_backtest import BacktestModels, EuropeanBacktester
from src.models.european_blend_search import BlendPoint, EuropeanBlendSweeper
from src.models.feature_engineer import FeatureEngineer
from src.models.poisson.dixon_coles import DixonColesModel


def _load_domestic(config) -> pd.DataFrame:
    """Every tracked league, cleaned and engineered as training does.

    Identical preparation matters even though only goals are used from here:
    the cleaner drops rows, and a dropped row is one the ELO walk never sees,
    so a shortcut would measure a rating history the served model never had.
    """
    loader = FootballDataLoader(config.data, hf_config=config.huggingface)
    frames = []
    for league_code in config.data.leagues:
        for season in config.data.seasons:
            df = loader.load_season(league_code, season)
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()

    data = DataCleaner().clean(pd.concat(frames, ignore_index=True))
    return FeatureEngineer(config.features).build_all_features(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the European blend weight and ELO draw rate"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Cap the number of holdout seasons (most recent kept)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    european_config = getattr(config, "european", None)
    if european_config is None or not european_config.enabled:
        print("european is disabled in config. Nothing to sweep.")
        sys.exit(0)
    if config.model.poisson is None or not config.model.poisson.enabled:
        print("model.poisson is disabled in config. Nothing to sweep.")
        sys.exit(0)

    prediction = european_config.prediction
    tuning = prediction.tuning

    print("\n=== Loading & preparing data ===")
    domestic = _load_domestic(config)
    if domestic.empty:
        print("ERROR: No domestic data loaded. Exiting.")
        sys.exit(1)
    supplementary = load_european_corpus(config, verbose=False)
    print(f"Domestic: {len(domestic)} matches   European: {len(supplementary)}")

    corpus = CrossCompetitionCorpus(domestic=domestic, supplementary=supplementary)
    backtester = EuropeanBacktester(
        corpus=corpus,
        prediction=prediction,
        models=BacktestModels(
            elo=lambda: FootballELO(config.features.elo),
            dixon_coles=lambda: DixonColesModel(config.model.poisson),
        ),
    )

    print(
        f"\n=== Backtest ({tuning.holdout_seasons} holdout seasons, "
        f"Dixon-Coles refitted per fold) ==="
    )
    dataset = backtester.run(max_folds=args.max_folds)
    for fold in dataset.folds:
        print(
            f"  {fold.label}: {fold.evaluated:>4} evaluated, "
            f"{fold.refused:>4} refused (cut {fold.cut.date()})"
        )
    if dataset.is_empty:
        print(
            "\nNo evaluable matches. Both teams must be in a tracked league and "
            "above european.prediction.min_matches_per_team."
        )
        sys.exit(1)
    print(f"  total: {len(dataset)} evaluated, {dataset.refused} refused")

    incumbent = BlendPoint(
        weight=prediction.dixon_coles_weight, draw_rate=prediction.elo_draw_rate
    )
    result = EuropeanBlendSweeper(tuning).sweep(dataset, incumbent)

    print("\n=== Baselines (held-out) ===")
    print(f"{'model':>18}  {'log_loss':>10}  {'brier':>8}  {'accuracy':>9}")
    for baseline in result.baselines:
        print(
            f"{baseline.name:>18}  {baseline.log_loss:>10.4f}  "
            f"{baseline.brier:>8.4f}  {baseline.accuracy:>9.3f}"
        )

    print("\n=== Best weight at each ELO draw rate ===")
    print(f"{'draw_rate':>10}  {'weight':>8}  {'log_loss':>10}")
    for draw_rate in tuning.draw_rate_grid:
        row = [c for c in result.cells if c.draw_rate == draw_rate]
        best_in_row = min(row, key=lambda cell: cell.log_loss)
        marker = "  <- best overall" if best_in_row == result.best else ""
        print(
            f"{draw_rate:>10.2f}  {best_in_row.weight:>8.2f}  "
            f"{best_in_row.log_loss:>10.4f}{marker}"
        )

    print(f"\n=== Weight profile at draw_rate {result.best.draw_rate:.2f} ===")
    print(f"{'weight':>8}  {'log_loss':>10}  {'brier':>8}  {'accuracy':>9}")
    for cell in [c for c in result.cells if c.draw_rate == result.best.draw_rate]:
        marker = "  <- best" if cell == result.best else ""
        print(
            f"{cell.weight:>8.2f}  {cell.log_loss:>10.4f}  "
            f"{cell.brier:>8.4f}  {cell.accuracy:>9.3f}{marker}"
        )

    low, high = result.improvement_interval
    print(
        f"\nBest: dixon_coles_weight {result.best.weight:.2f}, "
        f"elo_draw_rate {result.best.draw_rate:.2f} "
        f"(log-loss {result.best.log_loss:.4f})."
    )
    print(
        f"Current config: {incumbent.weight:.2f} / {incumbent.draw_rate:.2f}. "
        f"Improvement {result.improvement:+.4f} "
        f"[{low:+.4f}, {high:+.4f}] "
        f"at {tuning.confidence_level:.0%} over {result.matches} matches."
    )
    if result.is_conclusive:
        print(
            "The interval excludes zero: the improvement survives resampling. "
            "Report-only — update european.prediction in config.yaml to adopt."
        )
    else:
        print(
            "The interval spans zero: this measurement cannot distinguish the "
            "best cell from what is already configured. Leave it alone."
        )


if __name__ == "__main__":
    main()
