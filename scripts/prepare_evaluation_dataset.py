"""Evaluation dataset preparation — scores past predictions against actual results.

Reads past match predictions from the Supabase ``predictions`` table,
loads actual results from the football-data.co.uk historical CSVs,
scores each prediction set (accuracy + Brier score), and upserts
metrics to a Supabase ``evaluation_metrics`` table.

Usage:
    uv run python scripts/prepare_evaluation_dataset.py
    uv run python scripts/prepare_evaluation_dataset.py --days-back 14
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config


def _parse_date(date_str: str) -> datetime | None:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _fetch_past_predictions(supabase_client, days_back: int) -> list[dict]:
    oldest_dt = datetime.utcnow() - timedelta(days=days_back)
    cutoff_dt = datetime.utcnow() - timedelta(days=1)

    response = (
        supabase_client.table("predictions")
        .select("league_code, match_date, payload")
        .execute()
    )
    rows = response.data or []

    result = []
    for row in rows:
        parsed = _parse_date(row.get("match_date", ""))
        if parsed is not None and oldest_dt <= parsed <= cutoff_dt:
            result.append(row)
    return result


def _load_actual_results(config, league_code: str) -> dict[str, dict[str, str]]:
    """Returns {match_date_str: {"HomeTeam_vs_AwayTeam": "Home Win"|"Draw"|"Away Win"}}."""
    import pandas as pd

    from src.models.data_cleaner import DataCleaner
    from src.models.data_loader import FootballDataLoader

    loader = FootballDataLoader(config.data, hf_config=config.huggingface)
    frames = []
    for season in config.data.seasons:
        try:
            df = loader.load_season(league_code, season)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return {}

    cleaner = DataCleaner()
    data = cleaner.clean(pd.concat(frames, ignore_index=True))

    results: dict[str, dict[str, str]] = {}
    outcome_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

    for _, row in data.iterrows():
        date_val = row.get("Date", "")
        home = row.get("HomeTeam", "")
        away = row.get("AwayTeam", "")
        ftr = row.get("FTR", "")
        outcome = outcome_map.get(str(ftr), "")
        if not date_val or not home or not away or not outcome:
            continue

        date_str = str(date_val)
        if date_str not in results:
            results[date_str] = {}
        results[date_str][f"{home}_vs_{away}"] = outcome

    return results


def _resolve_actual(
    match_date: str, actual_by_date: dict[str, dict[str, str]]
) -> dict[str, str]:
    if match_date in actual_by_date:
        return actual_by_date[match_date]
    parsed = _parse_date(match_date)
    if parsed is None:
        return {}
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"):
        alt = parsed.strftime(fmt)
        if alt in actual_by_date:
            return actual_by_date[alt]
    return {}


def _score_predictions(
    matches: list[dict], actual_results: dict[str, str]
) -> dict[str, float | int]:
    correct = 0
    scored = 0
    brier_sum = 0.0

    for match in matches:
        key = f"{match.get('home_team', '')}_vs_{match.get('away_team', '')}"
        if key not in actual_results:
            continue
        actual = actual_results[key]
        scored += 1
        if match.get("predicted_outcome") == actual:
            correct += 1
        probs = match.get("probabilities", {})
        outcome_probs = {
            "Home Win": float(probs.get("home_win", 0.0)),
            "Draw": float(probs.get("draw", 0.0)),
            "Away Win": float(probs.get("away_win", 0.0)),
        }
        for outcome, prob in outcome_probs.items():
            indicator = 1.0 if outcome == actual else 0.0
            brier_sum += (prob - indicator) ** 2

    accuracy = correct / scored if scored > 0 else 0.0
    brier_score = brier_sum / scored if scored > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "brier_score": round(brier_score, 4),
        "scored_matches": scored,
        "total_predictions": len(matches),
    }


def _upsert_metrics(
    supabase_client, league_code: str, match_date: str, metrics: dict
) -> None:
    supabase_client.table("evaluation_metrics").upsert(
        {
            "league_code": league_code,
            "match_date": match_date,
            "accuracy": metrics["accuracy"],
            "brier_score": metrics["brier_score"],
            "scored_matches": metrics["scored_matches"],
            "total_predictions": metrics["total_predictions"],
            "evaluated_at": datetime.utcnow().isoformat(),
        },
        on_conflict="league_code,match_date",
    ).execute()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score past predictions against actual results"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=14,
        help="How many past days to evaluate (default: 14)",
    )
    args = parser.parse_args()

    from supabase import create_client

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

    client = create_client(url, key)
    config = load_config()

    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo = os.environ.get("HF_REPO_ID", "")
    if hf_token and hf_repo:
        from src.backend.services.model_loader import ModelLoader

        loader = ModelLoader(
            repo_id=hf_repo,
            hf_token=hf_token,
            local_dir=Path(os.environ.get("HF_LOCAL_DIR", "/tmp/hf_models")),
        )
        downloaded_path = loader.download()
        config.huggingface.local_dir = str(downloaded_path)

    print(f"Fetching past predictions (last {args.days_back} days)...")
    past_rows = _fetch_past_predictions(client, args.days_back)
    print(f"Found {len(past_rows)} prediction records to evaluate")

    if not past_rows:
        print("Nothing to evaluate.")
        return

    by_league: dict[str, list[dict]] = {}
    for row in past_rows:
        by_league.setdefault(row["league_code"], []).append(row)

    total_scored = 0
    for league_code, rows in by_league.items():
        print(f"\nLeague: {league_code}")
        actual_by_date = _load_actual_results(config, league_code)

        for row in rows:
            match_date = row["match_date"]
            matches = row.get("payload", {}).get("matches", [])
            actual = _resolve_actual(match_date, actual_by_date)

            if not actual:
                print(f"  {match_date}: no actual results found, skipping")
                continue

            metrics = _score_predictions(matches, actual)
            if metrics["scored_matches"] == 0:
                print(f"  {match_date}: 0 matches scored, skipping")
                continue

            _upsert_metrics(client, league_code, match_date, metrics)
            print(
                f"  {match_date}: accuracy={metrics['accuracy']:.1%}, "
                f"brier={metrics['brier_score']:.4f}, "
                f"scored={metrics['scored_matches']}/{metrics['total_predictions']}"
            )
            total_scored += metrics["scored_matches"]

    print(f"\nDone. Scored {total_scored} matches total.")


if __name__ == "__main__":
    main()
