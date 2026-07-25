"""
Offline inference runner — computes predictions for all leagues and uploads
the results to a Supabase ``predictions`` table.

Designed to run in GitHub Actions (where memory/time are not constrained).
The backend then reads from Supabase instead of running the ML pipeline.

Usage:
    uv run python scripts/run_inference.py
    uv run python scripts/run_inference.py --date 26/04/2026
    uv run python scripts/run_inference.py --leagues E0,D1
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from src.analysis.match_stats import MatchStatsCalculator
from src.analysis.value_detector import ValueDetector
from src.models.data_cleaner import DataCleaner
from src.models.data_loader import FootballDataLoader
from src.models.elo import FootballELO
from src.models.feature_engineer import FeatureEngineer
from src.models.predictor import MatchPrediction, MatchPredictor
from src.scrapers.fixtures_fetcher import (
    Fixture,
    fetch_available_dates,
    fetch_fixtures,
)
from src.scrapers.odds_aggregator import AggregatedOdds
from src.scrapers.base_scraper import ScrapedOdds

import pandas as pd


# ---------------------------------------------------------------------------
# Prediction logic (mirrors PredictionService._compute_predictions)
# ---------------------------------------------------------------------------


def _load_league_featured_data(config, league_code: str) -> pd.DataFrame:
    loader = FootballDataLoader(config.data, hf_config=config.huggingface)
    frames = []
    for season in config.data.seasons:
        df = loader.load_season(league_code, season)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    del frames

    cleaner = DataCleaner()
    data = cleaner.clean(data)

    engineer = FeatureEngineer(config.features)
    data = engineer.build_all_features(data)

    if data.empty:
        return data

    elo = FootballELO(config.features.elo)
    data = elo.compute_elo_features(data)
    return data


def _predict_from_odds(fixture: Fixture) -> MatchPrediction:
    probs = fixture.implied_probabilities()
    if all(v == 0 for v in probs.values()):
        probs = {"home": 0.40, "draw": 0.25, "away": 0.35}
    pred_outcome = max(probs, key=lambda k: probs[k])
    outcome_map = {"home": "Home Win", "draw": "Draw", "away": "Away Win"}
    return MatchPrediction(
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        home_win_prob=probs["home"],
        draw_prob=probs["draw"],
        away_win_prob=probs["away"],
        predicted_outcome=outcome_map.get(pred_outcome, "Unknown"),
        confidence=max(probs.values()),
    )


def _predict_fixture(fixture, featured_data, predictor):
    """Re-uses the PredictionService logic for a single fixture."""
    home = fixture.home_team
    away = fixture.away_team

    if featured_data.empty:
        return _predict_from_odds(fixture)

    home_rows = featured_data[
        (featured_data["HomeTeam"] == home) | (featured_data["AwayTeam"] == home)
    ]
    away_rows = featured_data[
        (featured_data["HomeTeam"] == away) | (featured_data["AwayTeam"] == away)
    ]

    league_avg = featured_data.mean(numeric_only=True)
    home_from_avg = home_rows.empty
    away_from_avg = away_rows.empty
    last_home_row = home_rows.iloc[-1] if not home_from_avg else league_avg
    last_away_row = away_rows.iloc[-1] if not away_from_avg else league_avg

    # When using league averages, treat as if the team is playing in expected position
    home_was_home = home_from_avg or last_home_row.get("HomeTeam", "") == home
    away_was_away = away_from_avg or last_away_row.get("AwayTeam", "") == away

    def _get(row, was_expected, expected, opposite):
        val = row.get(expected if was_expected else opposite, 0)
        return val if pd.notna(val) else 0

    feature_row: dict = {}
    for feat in predictor.feature_names:
        if feat.startswith("home_"):
            s = feat[len("home_"):]
            feature_row[feat] = _get(last_home_row, home_was_home, f"home_{s}", f"away_{s}")
        elif feat.startswith("away_"):
            s = feat[len("away_"):]
            feature_row[feat] = _get(last_away_row, away_was_away, f"away_{s}", f"home_{s}")
        elif feat.startswith("diff_"):
            s = feat[len("diff_"):]
            h = _get(last_home_row, home_was_home, f"home_{s}", f"away_{s}")
            a = _get(last_away_row, away_was_away, f"away_{s}", f"home_{s}")
            feature_row[feat] = h - a
        elif feat == "elo_home":
            feature_row[feat] = _get(last_home_row, home_was_home, "elo_home", "elo_away")
        elif feat == "elo_away":
            feature_row[feat] = _get(last_away_row, away_was_away, "elo_away", "elo_home")
        elif feat == "elo_diff":
            h = _get(last_home_row, home_was_home, "elo_home", "elo_away")
            a = _get(last_away_row, away_was_away, "elo_away", "elo_home")
            feature_row[feat] = h - a
        elif feat == "elo_expected_home":
            feature_row[feat] = _get(last_home_row, home_was_home, "elo_expected_home", "elo_expected_away")
        elif feat == "elo_expected_away":
            feature_row[feat] = _get(last_away_row, away_was_away, "elo_expected_away", "elo_expected_home")
        elif feat == "xG_diff":
            h = _get(last_home_row, home_was_home, "home_xG_rolling", "away_xG_rolling")
            a = _get(last_away_row, away_was_away, "away_xG_rolling", "home_xG_rolling")
            feature_row[feat] = h - a
        elif feat == "rest_advantage":
            h = _get(last_home_row, home_was_home, "home_rest_days", "away_rest_days")
            a = _get(last_away_row, away_was_away, "away_rest_days", "home_rest_days")
            feature_row[feat] = h - a
        elif feat == "avg_draw_pct":
            h = _get(last_home_row, home_was_home, "home_draw_pct", "away_draw_pct")
            a = _get(last_away_row, away_was_away, "away_draw_pct", "home_draw_pct")
            feature_row[feat] = (h + a) / 2
        elif feat == "form_gap":
            h = _get(last_home_row, home_was_home, "home_Form", "away_Form")
            a = _get(last_away_row, away_was_away, "away_Form", "home_Form")
            feature_row[feat] = abs(h - a)
        elif feat == "attack_similarity":
            h = _get(last_home_row, home_was_home, "home_avg_GF", "away_avg_GF")
            a = _get(last_away_row, away_was_away, "away_avg_GF", "home_avg_GF")
            feature_row[feat] = 1 / (1 + abs(h - a))
        elif feat == "defense_similarity":
            h = _get(last_home_row, home_was_home, "home_avg_GA", "away_avg_GA")
            a = _get(last_away_row, away_was_away, "away_avg_GA", "home_avg_GA")
            feature_row[feat] = 1 / (1 + abs(h - a))
        elif feat == "combined_defensive":
            h = _get(last_home_row, home_was_home, "home_avg_GA", "away_avg_GA")
            a = _get(last_away_row, away_was_away, "away_avg_GA", "home_avg_GA")
            feature_row[feat] = 1 / (1 + h) + 1 / (1 + a)
        elif feat == "is_midweek":
            feature_row[feat] = 0
        elif feat.startswith("h2h_"):
            h2h_rows = featured_data[
                ((featured_data["HomeTeam"] == home) & (featured_data["AwayTeam"] == away))
                | ((featured_data["HomeTeam"] == away) & (featured_data["AwayTeam"] == home))
            ]
            if not h2h_rows.empty:
                h2h_last = h2h_rows.iloc[-1]
                h2h_was_home = h2h_last.get("HomeTeam") == home
                if feat == "h2h_home_wins" and not h2h_was_home:
                    val = h2h_last.get(feat, 0)
                    val = 0 if pd.isna(val) else val
                    draws_val = h2h_last.get("h2h_draws", 0)
                    draws_val = 0 if pd.isna(draws_val) else draws_val
                    feature_row[feat] = max(0, 1 - val - draws_val)
                else:
                    val = h2h_last.get(feat, 0)
                    feature_row[feat] = 0 if pd.isna(val) else val
            else:
                feature_row[feat] = 0
        else:
            val = last_home_row.get(feat, 0)
            feature_row[feat] = 0 if pd.isna(val) else val

    feature_row["HomeTeam"] = home
    feature_row["AwayTeam"] = away

    match_df = pd.DataFrame([feature_row])
    predictions = predictor.predict(match_df)
    return predictions[0] if predictions else _predict_from_odds(fixture)


# ---------------------------------------------------------------------------
# Build the API-compatible response payload (matches LeaguePredictionsResponse)
# ---------------------------------------------------------------------------


def _build_response_payload(
    league_code: str,
    league_name: str,
    fixtures: list[Fixture],
    predictions: list[MatchPrediction],
    match_stats: list,
    value_bets: list,
) -> dict:
    """Build the exact JSON shape that the API returns."""
    matches = []
    for fixture, pred, stats in zip(fixtures, predictions, match_stats):
        match_vbs = [
            vb for vb in value_bets
            if vb.home_team == pred.home_team and vb.away_team == pred.away_team
        ]

        if fixture.has_odds:
            bk_probs = fixture.implied_probabilities()
            odds_field: dict | None = {
                "home": fixture.b365_home,
                "draw": fixture.b365_draw,
                "away": fixture.b365_away,
            }
            implied_field: dict | None = {
                "home": round(bk_probs["home"], 4),
                "draw": round(bk_probs["draw"], 4),
                "away": round(bk_probs["away"], 4),
            }
        else:
            odds_field = None
            implied_field = None

        match_data: dict = {
            "home_team": pred.home_team,
            "away_team": pred.away_team,
            "league": fixture.league,
            "time": fixture.time or "",
            "probabilities": {
                "home_win": round(pred.home_win_prob, 4),
                "draw": round(pred.draw_prob, 4),
                "away_win": round(pred.away_win_prob, 4),
            },
            "predicted_outcome": pred.predicted_outcome,
            "confidence": round(pred.confidence, 4),
            "odds": odds_field,
            "implied_probabilities": implied_field,
            "expected_goals": None,
            "over_under": None,
            "btts": None,
            "top_scorelines": None,
            "form": None,
            "value_bets": [
                {
                    "outcome": vb.outcome,
                    "ml_probability": round(vb.ml_probability, 4),
                    "bookmaker_implied": round(vb.bookmaker_probability, 4),
                    "edge": round(vb.edge, 4),
                    "edge_pct": f"{vb.edge * 100:.1f}%",
                    "best_odds": round(vb.best_odds, 2),
                    "kelly_fraction": round(vb.kelly_fraction, 4),
                    "confidence": vb.confidence,
                }
                for vb in match_vbs
            ],
        }

        if stats:
            match_data["expected_goals"] = {
                "home": round(stats.home_xg, 2),
                "away": round(stats.away_xg, 2),
                "total": round(stats.total_xg, 2),
            }
            match_data["over_under"] = {
                "over_15": round(stats.over15_prob, 3),
                "over_25": round(stats.over25_prob, 3),
                "over_35": round(stats.over35_prob, 3),
                "under_25": round(stats.under25_prob, 3),
            }
            match_data["btts"] = {
                "yes": round(stats.btts_yes_prob, 3),
                "no": round(stats.btts_no_prob, 3),
            }
            match_data["top_scorelines"] = [
                {"score": f"{h}-{a}", "prob": round(p, 3)}
                for h, a, p in (stats.top_scorelines or [])[:5]
            ]
            match_data["form"] = {
                "home": round(stats.home_form, 2),
                "away": round(stats.away_form, 2),
            }

        matches.append(match_data)

    return {
        "league_code": league_code,
        "league_name": league_name,
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Upload to Supabase
# ---------------------------------------------------------------------------


def _upload_to_supabase(league_code: str, match_date: str, payload: dict) -> None:
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

    client = create_client(url, key)
    client.table("predictions").upsert(
        {
            "league_code": league_code,
            "match_date": match_date,
            "payload": payload,
        },
        on_conflict="league_code,match_date",
    ).execute()
    print(f"  Uploaded {league_code} / {match_date} to Supabase")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML inference and upload to Supabase")
    parser.add_argument("--date", default=None, help="Target date DD/MM/YYYY (default: all fixture dates)")
    parser.add_argument("--leagues", default=None, help="Comma-separated league codes (default: all)")
    args = parser.parse_args()

    config = load_config()

    # Resolve leagues
    league_codes = (
        [lc.strip().upper() for lc in args.leagues.split(",")]
        if args.leagues
        else list(config.data.leagues.keys())
    )

    # Resolve dates
    if args.date:
        target_dates = [args.date]
    else:
        target_dates = fetch_available_dates(league_codes)
        today = datetime.now().strftime("%d/%m/%Y")
        if today not in target_dates:
            target_dates.append(today)
        target_dates = sorted(target_dates, key=lambda d: datetime.strptime(d, "%d/%m/%Y"))

    if not target_dates:
        print("No fixture dates found.")
        return

    print(f"Leagues: {league_codes}")
    print(f"Dates:   {target_dates}")

    # Load model
    model_dir = Path(config.output.models_dir)
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
        model_path = loader.get_model_path("ensemble_model.joblib")
        model_dir = model_path.parent
        config.output.models_dir = str(model_dir)
        config.huggingface.local_dir = str(downloaded_path)
        print(f"Model downloaded from HF to {model_dir}")

    predictor: MatchPredictor | None = None
    model_file = model_dir / "ensemble_model.joblib"
    if model_file.exists():
        predictor = MatchPredictor(str(model_dir))
        print("ML model loaded.")
    else:
        print("WARNING: No model found, will use odds-based fallback.")

    # Reuse the predictor's Dixon-Coles model (when present) so the extended
    # markets are derived from the calibrated scoreline distribution rather
    # than the naive independent-Poisson fallback.
    poisson_model = predictor.poisson if predictor is not None else None
    stats_calc = MatchStatsCalculator(config.data, poisson_model=poisson_model)
    detector = ValueDetector(config.analysis)

    total_uploaded = 0

    for league_code in league_codes:
        league_name = config.data.leagues.get(league_code, league_code)
        print(f"\n{'='*60}")
        print(f"League: {league_name} ({league_code})")

        # Load featured data ONCE per league
        featured_data = pd.DataFrame()
        if predictor:
            print(f"  Loading historical data...")
            featured_data = _load_league_featured_data(config, league_code)
            print(f"  Historical data: {len(featured_data)} rows")

        for target_date in target_dates:
            fixtures = fetch_fixtures(target_date=target_date, leagues=[league_code])
            if not fixtures:
                continue

            print(f"  Date {target_date}: {len(fixtures)} fixtures")

            # Predict
            predictions = []
            odds_list = []
            for fixture in fixtures:
                if predictor and not featured_data.empty:
                    pred = _predict_fixture(fixture, featured_data, predictor)
                else:
                    pred = _predict_from_odds(fixture)
                predictions.append(pred)
                # Only build odds entries (and thus value bets) when the
                # fixture actually has Bet365 prices; otherwise it is N/A.
                if not fixture.has_odds:
                    continue
                scraped = ScrapedOdds(
                    source="Bet365",
                    home_team=fixture.home_team,
                    away_team=fixture.away_team,
                    home_win=fixture.b365_home,
                    draw=fixture.b365_draw,
                    away_win=fixture.b365_away,
                    league=fixture.league,
                    match_date=fixture.date,
                    url="",
                )
                odds_list.append(AggregatedOdds(
                    home_team=fixture.home_team,
                    away_team=fixture.away_team,
                    league=fixture.league,
                    match_date=fixture.date,
                    sources=[scraped],
                ))

            # Match stats
            match_stats = []
            for fixture in fixtures:
                try:
                    stats = stats_calc.compute_match_stats(
                        fixture.division, fixture.home_team, fixture.away_team
                    )
                    match_stats.append(stats)
                except Exception:
                    match_stats.append(None)

            # Value bets
            value_bets = detector.find_value_bets_batch(predictions, odds_list)

            # Build response payload
            payload = _build_response_payload(
                league_code, league_name, fixtures, predictions, match_stats, value_bets
            )

            # Upload
            _upload_to_supabase(league_code, target_date, payload)
            total_uploaded += 1

        # Free memory between leagues
        del featured_data
        gc.collect()

    print(f"\nDone. Uploaded {total_uploaded} prediction sets to Supabase.")


if __name__ == "__main__":
    main()
