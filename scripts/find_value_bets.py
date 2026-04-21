"""
Find value bets by combining ML model predictions with bookmaker odds.

Instead of scraping betting sites, this script:
1. Fetches fixtures + B365 odds from football-data.co.uk
2. Loads the trained ML model and predicts match probabilities
3. Compares ML vs bookmaker probabilities
4. Tells you the minimum odds to look for on any betting site

Usage:
    uv run python scripts/find_value_bets.py --matches "Leverkusen vs Augsburg, Chelsea vs Man United"
    uv run python scripts/find_value_bets.py --league E0
    uv run python scripts/find_value_bets.py --league D1 --date 18/04/2026
    uv run python scripts/find_value_bets.py --manual "Ath Madrid vs Sociedad" --manual-league SP1 --manual-odds "1.80/3.50/4.50"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from src.analysis.match_stats import MatchStats, MatchStatsCalculator
from src.analysis.value_detector import ValueBet, ValueDetector
from src.models.predictor import MatchPrediction
from src.scrapers.base_scraper import ScrapedOdds
from src.scrapers.fixtures_fetcher import Fixture, fetch_fixtures, fetch_fixtures_for_matches
from src.scrapers.odds_aggregator import AggregatedOdds
from src.visualization.html_report import generate_html_report


def fixture_to_aggregated_odds(fixture: Fixture) -> AggregatedOdds:
    """Convert a Fixture (CSV row) into an AggregatedOdds object."""
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
    return AggregatedOdds(
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        league=fixture.league,
        match_date=fixture.date,
        sources=[scraped],
    )


def make_prediction_from_fixture(
    fixture: Fixture, config: object = None
) -> MatchPrediction:
    """
    Try to predict using the trained ML model.
    Falls back to B365 implied probabilities if no model is available.
    """
    if config is None:
        config = load_config()
    model_dir = Path(config.output.models_dir)

    if model_dir.exists() and (model_dir / "ensemble_model.joblib").exists():
        return _predict_with_model(fixture, model_dir, config)
    else:
        return _predict_from_odds(fixture)


def _predict_with_model(fixture: Fixture, model_dir: Path, config) -> MatchPrediction:
    """Predict using the trained ML ensemble model."""
    from src.models.data_cleaner import DataCleaner
    from src.models.data_loader import FootballDataLoader
    from src.models.elo import FootballELO
    from src.models.feature_engineer import FeatureEngineer
    from src.models.predictor import MatchPredictor

    # Load historical data for the league
    league_code = fixture.division
    loader = FootballDataLoader(config.data)

    frames = []
    for season in config.data.seasons:
        df = loader.load_season(league_code, season)
        if not df.empty:
            frames.append(df)

    if not frames:
        print(f"    No historical data for {fixture.league}, falling back to odds")
        return _predict_from_odds(fixture)

    import pandas as pd
    data = pd.concat(frames, ignore_index=True)

    # Clean + feature engineer (same pipeline as training)
    cleaner = DataCleaner()
    data = cleaner.clean(data)

    engineer = FeatureEngineer(config.features)
    featured_data = engineer.build_all_features(data)

    if featured_data.empty:
        print(f"    No features built for {fixture.league}, falling back to odds")
        return _predict_from_odds(fixture)

    # Add ELO
    elo = FootballELO(config.features.elo)
    featured_data = elo.compute_elo_features(featured_data)

    # Get latest stats for each team by finding last row where they played
    home = fixture.home_team
    away = fixture.away_team

    home_rows = featured_data[
        (featured_data["HomeTeam"] == home) | (featured_data["AwayTeam"] == home)
    ]
    away_rows = featured_data[
        (featured_data["HomeTeam"] == away) | (featured_data["AwayTeam"] == away)
    ]

    if home_rows.empty or away_rows.empty:
        print(f"    Teams not found in historical data, falling back to odds")
        return _predict_from_odds(fixture)

    # Build a synthetic match row from the last known stats
    # Take the most recent match in the dataset as a template
    last_home_row = home_rows.iloc[-1]
    last_away_row = away_rows.iloc[-1]

    # CRITICAL: Determine if each team was home or away in their last match.
    # Features like home_avg_GF represent the HOME team's stats in that row.
    # If our prediction's home team was AWAY in their last match, we must read
    # away_* features from that row (those are the team's own stats).
    home_was_home = last_home_row.get("HomeTeam") == home
    away_was_away = last_away_row.get("AwayTeam") == away

    def _get_team_val(row, was_in_expected_role, feat_expected, feat_opposite):
        """Get a team's own stat from a match row, swapping prefixes if needed."""
        val = row.get(feat_expected if was_in_expected_role else feat_opposite, 0)
        return val if pd.notna(val) else 0

    predictor = MatchPredictor(str(model_dir))

    # Build feature vector matching the model's expected features.
    # For each feature, extract the correct team's value by checking whether
    # the team was home or away in the row we're reading from.
    feature_row = {}
    for feat in predictor.feature_names:
        if feat.startswith("home_"):
            suffix = feat[len("home_"):]
            feature_row[feat] = _get_team_val(
                last_home_row, home_was_home, f"home_{suffix}", f"away_{suffix}"
            )
        elif feat.startswith("away_"):
            suffix = feat[len("away_"):]
            feature_row[feat] = _get_team_val(
                last_away_row, away_was_away, f"away_{suffix}", f"home_{suffix}"
            )
        elif feat.startswith("diff_"):
            suffix = feat[len("diff_"):]
            h_val = _get_team_val(
                last_home_row, home_was_home, f"home_{suffix}", f"away_{suffix}"
            )
            a_val = _get_team_val(
                last_away_row, away_was_away, f"away_{suffix}", f"home_{suffix}"
            )
            feature_row[feat] = h_val - a_val
        elif feat == "elo_home":
            feature_row[feat] = _get_team_val(
                last_home_row, home_was_home, "elo_home", "elo_away"
            )
        elif feat == "elo_away":
            feature_row[feat] = _get_team_val(
                last_away_row, away_was_away, "elo_away", "elo_home"
            )
        elif feat == "elo_diff":
            h_elo = _get_team_val(
                last_home_row, home_was_home, "elo_home", "elo_away"
            )
            a_elo = _get_team_val(
                last_away_row, away_was_away, "elo_away", "elo_home"
            )
            feature_row[feat] = h_elo - a_elo
        elif feat == "elo_expected_home":
            feature_row[feat] = _get_team_val(
                last_home_row, home_was_home, "elo_expected_home", "elo_expected_away"
            )
        elif feat == "elo_expected_away":
            feature_row[feat] = _get_team_val(
                last_away_row, away_was_away, "elo_expected_away", "elo_expected_home"
            )
        elif feat == "xG_diff":
            h_xg = _get_team_val(
                last_home_row, home_was_home, "home_xG_rolling", "away_xG_rolling"
            )
            a_xg = _get_team_val(
                last_away_row, away_was_away, "away_xG_rolling", "home_xG_rolling"
            )
            feature_row[feat] = h_xg - a_xg
        elif feat == "rest_advantage":
            h_rest = _get_team_val(
                last_home_row, home_was_home, "home_rest_days", "away_rest_days"
            )
            a_rest = _get_team_val(
                last_away_row, away_was_away, "away_rest_days", "home_rest_days"
            )
            feature_row[feat] = h_rest - a_rest
        elif feat == "avg_draw_pct":
            h_dp = _get_team_val(
                last_home_row, home_was_home, "home_draw_pct", "away_draw_pct"
            )
            a_dp = _get_team_val(
                last_away_row, away_was_away, "away_draw_pct", "home_draw_pct"
            )
            feature_row[feat] = (h_dp + a_dp) / 2
        elif feat == "form_gap":
            h_form = _get_team_val(
                last_home_row, home_was_home, "home_Form", "away_Form"
            )
            a_form = _get_team_val(
                last_away_row, away_was_away, "away_Form", "home_Form"
            )
            feature_row[feat] = abs(h_form - a_form)
        elif feat == "attack_similarity":
            h_gf = _get_team_val(
                last_home_row, home_was_home, "home_avg_GF", "away_avg_GF"
            )
            a_gf = _get_team_val(
                last_away_row, away_was_away, "away_avg_GF", "home_avg_GF"
            )
            feature_row[feat] = 1 / (1 + abs(h_gf - a_gf))
        elif feat == "defense_similarity":
            h_ga = _get_team_val(
                last_home_row, home_was_home, "home_avg_GA", "away_avg_GA"
            )
            a_ga = _get_team_val(
                last_away_row, away_was_away, "away_avg_GA", "home_avg_GA"
            )
            feature_row[feat] = 1 / (1 + abs(h_ga - a_ga))
        elif feat == "combined_defensive":
            h_ga = _get_team_val(
                last_home_row, home_was_home, "home_avg_GA", "away_avg_GA"
            )
            a_ga = _get_team_val(
                last_away_row, away_was_away, "away_avg_GA", "home_avg_GA"
            )
            feature_row[feat] = 1 / (1 + h_ga) + 1 / (1 + a_ga)
        elif feat == "is_midweek":
            feature_row[feat] = 0
        elif feat.startswith("h2h_"):
            # H2H features are relative to the home team of the row.
            # If our home team was away in that row, h2h_home_wins is
            # actually the opponent's H2H win rate, so we need to look
            # at the last row where both teams met instead.
            # For now, find the last row where BOTH teams played each other.
            h2h_rows = featured_data[
                ((featured_data["HomeTeam"] == home) & (featured_data["AwayTeam"] == away))
                | ((featured_data["HomeTeam"] == away) & (featured_data["AwayTeam"] == home))
            ]
            if not h2h_rows.empty:
                h2h_last = h2h_rows.iloc[-1]
                h2h_was_home = h2h_last.get("HomeTeam") == home
                if feat == "h2h_home_wins" and not h2h_was_home:
                    # Opponent's H2H win rate from that row — invert
                    val = h2h_last.get(feat, 0)
                    val = 0 if pd.isna(val) else val
                    draws_val = h2h_last.get("h2h_draws", 0)
                    draws_val = 0 if pd.isna(draws_val) else draws_val
                    feature_row[feat] = max(0, 1 - val - draws_val)
                elif h2h_was_home:
                    val = h2h_last.get(feat, 0)
                    feature_row[feat] = 0 if pd.isna(val) else val
                else:
                    val = h2h_last.get(feat, 0)
                    feature_row[feat] = 0 if pd.isna(val) else val
            else:
                feature_row[feat] = 0
        elif feat.startswith("home_") or feat.startswith("away_"):
            # Catch-all for any home_/away_ prefixed features not handled above
            if feat.startswith("home_"):
                suffix = feat[len("home_"):]
                feature_row[feat] = _get_team_val(
                    last_home_row, home_was_home, f"home_{suffix}", f"away_{suffix}"
                )
            else:
                suffix = feat[len("away_"):]
                feature_row[feat] = _get_team_val(
                    last_away_row, away_was_away, f"away_{suffix}", f"home_{suffix}"
                )
        else:
            feature_row[feat] = last_home_row.get(feat, 0)
            if pd.isna(feature_row[feat]):
                feature_row[feat] = 0

    # Add team identifiers for the prediction output
    feature_row["HomeTeam"] = home
    feature_row["AwayTeam"] = away

    match_df = pd.DataFrame([feature_row])
    predictions = predictor.predict(match_df)

    if predictions:
        return predictions[0]

    return _predict_from_odds(fixture)


def _predict_from_odds(fixture: Fixture) -> MatchPrediction:
    """Fallback: derive prediction from B365 implied probabilities."""
    probs = fixture.implied_probabilities()
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


def compute_min_value_odds(ml_prob: float) -> float:
    """
    Given ML probability, compute minimum odds that represent value.
    Value exists when: 1/odds < ml_prob  →  odds > 1/ml_prob
    """
    if ml_prob <= 0:
        return float("inf")
    return round(1 / ml_prob, 2)


def parse_matches(matches_str: str) -> list[tuple[str, str]]:
    """Parse 'Home vs Away, Home2 vs Away2' into list of tuples."""
    matches = []
    for match in matches_str.split(","):
        parts = match.strip().split(" vs ")
        if len(parts) == 2:
            matches.append((parts[0].strip(), parts[1].strip()))
        else:
            print(f"  Warning: Could not parse match '{match.strip()}'")
    return matches


def create_manual_fixtures(
    matches_str: str,
    league_code: str,
    odds_str: str | None = None,
) -> list[Fixture]:
    """Create Fixture objects from manual input for cup/non-league games."""
    from src.scrapers.fixtures_fetcher import DIVISION_MAP
    from datetime import datetime

    match_list = parse_matches(matches_str)
    league_name = DIVISION_MAP.get(league_code, league_code)
    today = datetime.now().strftime("%d/%m/%Y")

    # Parse odds: "H/D/A" or "H/D/A, H/D/A, ..."
    odds_sets: list[tuple[float, float, float]] = []
    if odds_str:
        for odds_group in odds_str.split(","):
            parts = odds_group.strip().split("/")
            if len(parts) == 3:
                odds_sets.append((float(parts[0]), float(parts[1]), float(parts[2])))

    fixtures = []
    for i, (home, away) in enumerate(match_list):
        if i < len(odds_sets):
            h, d, a = odds_sets[i]
        else:
            h, d, a = 0.0, 0.0, 0.0

        fixtures.append(Fixture(
            division=league_code,
            league=league_name,
            date=today,
            time="",
            home_team=home,
            away_team=away,
            b365_home=h,
            b365_draw=d,
            b365_away=a,
        ))

    return fixtures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find value bets: ML predictions vs bookmaker odds"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--matches", default=None,
        help='Specific matches, comma-separated: "Home vs Away, Home2 vs Away2"',
    )
    parser.add_argument(
        "--league", default=None,
        help="Division code(s), comma-separated (e.g. E0,D1)",
    )
    parser.add_argument(
        "--date", default=None,
        help="Date in DD/MM/YYYY format. Defaults to today.",
    )
    parser.add_argument(
        "--output", default=None, help="Output report filename"
    )
    parser.add_argument(
        "--manual", default=None,
        help='Manual matches (cup games etc): "Home vs Away, Home2 vs Away2"',
    )
    parser.add_argument(
        "--manual-league", default="SP1",
        help="League code for manual matches (for historical data lookup). Default: SP1",
    )
    parser.add_argument(
        "--manual-odds", default=None,
        help='Odds for manual matches: "H/D/A" or "H/D/A, H/D/A" (comma-separated per match)',
    )
    parser.add_argument(
        "--export", default=None, choices=["csv", "excel"],
        help="Export predictions to file: csv or excel",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Step 1: Fetch fixtures
    print("\n=== Step 1: Fetching Fixtures ===")
    fixtures: list[Fixture] = []

    if args.manual:
        manual_fixtures = create_manual_fixtures(
            args.manual, args.manual_league, args.manual_odds
        )
        fixtures.extend(manual_fixtures)
        print(f"Added {len(manual_fixtures)} manual match(es)")

    if args.matches:
        match_list = parse_matches(args.matches)
        csv_fixtures = fetch_fixtures_for_matches(match_list, target_date=args.date)
        fixtures.extend(csv_fixtures)
        print(f"Requested {len(match_list)} matches from CSV, found {len(csv_fixtures)}")
    elif args.league:
        leagues = [l.strip() for l in args.league.split(",")]
        csv_fixtures = fetch_fixtures(target_date=args.date, leagues=leagues)
        fixtures.extend(csv_fixtures)
        print(f"Found {len(csv_fixtures)} fixtures from CSV")
    elif not args.manual:
        # Default: all supported leagues
        supported = list(config.data.leagues.keys())
        csv_fixtures = fetch_fixtures(target_date=args.date, leagues=supported)
        fixtures.extend(csv_fixtures)
        print(f"Found {len(csv_fixtures)} fixtures across supported leagues")

    print(f"Total fixtures: {len(fixtures)}")

    if not fixtures:
        print("No fixtures found. Try --date or --league options.")
        sys.exit(0)

    # Step 2: Generate predictions
    print("\n=== Step 2: Generating ML Predictions ===")
    predictions: list[MatchPrediction] = []
    odds_list: list[AggregatedOdds] = []

    for fixture in fixtures:
        print(f"  Predicting: {fixture.home_team} vs {fixture.away_team}...")
        pred = make_prediction_from_fixture(fixture, config)
        predictions.append(pred)
        odds_list.append(fixture_to_aggregated_odds(fixture))

    # Step 3: Compute extended match stats (Poisson model)
    print(f"\n=== Step 3: Computing Extended Stats ===")
    stats_calc = MatchStatsCalculator(config.data)
    match_stats_list: list[MatchStats | None] = []

    for fixture in fixtures:
        if fixture.division and fixture.division in config.data.leagues:
            print(f"  Stats: {fixture.home_team} vs {fixture.away_team}...")
            stats = stats_calc.compute_match_stats(
                fixture.division, fixture.home_team, fixture.away_team
            )
            match_stats_list.append(stats)
        else:
            match_stats_list.append(None)

    # Step 4: Find value bets
    print(f"\n=== Step 4: Value Bet Analysis ===")
    detector = ValueDetector(config.analysis)
    value_bets = detector.find_value_bets_batch(predictions, odds_list)

    # Step 5: Display results
    print("\n" + "=" * 70)
    print("  MATCH ANALYSIS & VALUE BETS")
    print("=" * 70)

    for pred, fixture, stats in zip(predictions, fixtures, match_stats_list):
        bk_probs = fixture.implied_probabilities()
        print(f"\n  {fixture.time:5s}  {pred.home_team} vs {pred.away_team} ({fixture.league})")
        print(f"  {'':5s}  B365 Odds: {fixture.b365_home:.2f} / {fixture.b365_draw:.2f} / {fixture.b365_away:.2f}")
        print(f"  {'':5s}  B365 Implied:  H={bk_probs['home']:.1%}  D={bk_probs['draw']:.1%}  A={bk_probs['away']:.1%}")
        print(f"  {'':5s}  ML Predicted:  H={pred.home_win_prob:.1%}  D={pred.draw_prob:.1%}  A={pred.away_win_prob:.1%}")
        print(f"  {'':5s}  Prediction: {pred.predicted_outcome} ({pred.confidence:.1%})")

        if stats:
            print(f"  {'':5s}  xG: {stats.home_xg:.1f} - {stats.away_xg:.1f} (total: {stats.total_xg:.1f})")
            print(f"  {'':5s}  Over 2.5: {stats.over25_prob:.0%} | BTTS: {stats.btts_yes_prob:.0%}")
            top = stats.top_scorelines[:3]
            scores_str = ", ".join(f"{h}-{a} ({p:.0%})" for h, a, p in top)
            print(f"  {'':5s}  Top scores: {scores_str}")

        # Show minimum value odds for each outcome
        min_h = compute_min_value_odds(pred.home_win_prob)
        min_d = compute_min_value_odds(pred.draw_prob)
        min_a = compute_min_value_odds(pred.away_win_prob)
        print(f"  {'':5s}  Look for odds ≥  H: {min_h:.2f}  D: {min_d:.2f}  A: {min_a:.2f}")

        # Show specific value bets for this match
        match_vbs = [
            vb for vb in value_bets
            if vb.home_team == pred.home_team and vb.away_team == pred.away_team
        ]
        if match_vbs:
            for vb in match_vbs:
                print(
                    f"  {'':5s}  >>> VALUE: {vb.outcome} | "
                    f"Edge: {vb.edge:.1%} | "
                    f"Kelly: {vb.kelly_fraction:.1%} | "
                    f"Confidence: {vb.confidence}"
                )

    # Summary
    if value_bets:
        print("\n" + "-" * 70)
        print("  VALUE BETS SUMMARY (sorted by edge)")
        print("-" * 70)
        for vb in value_bets:
            print(
                f"  {vb.home_team} vs {vb.away_team} → {vb.outcome}"
            )
            print(
                f"    ML: {vb.ml_probability:.1%} vs B365: {vb.bookmaker_probability:.1%} | "
                f"Edge: {vb.edge:.1%} | Best odds: {vb.best_odds:.2f} | "
                f"Kelly: {vb.kelly_fraction:.1%} | {vb.confidence}"
            )
    else:
        print("\n  No value bets detected with current thresholds.")

    print("\n" + "=" * 70)

    # Save JSON report
    json_path = args.output or f"{config.output.reports_dir}/value_report_latest.json"
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "predictions": [p.to_dict() for p in predictions],
        "fixtures": [f.to_dict() for f in fixtures],
        "match_stats": [s.to_dict() if s else None for s in match_stats_list],
        "value_bets": [v.to_dict() for v in value_bets],
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON report saved to {json_path}")

    # Generate HTML report
    html_path = Path(config.output.reports_dir) / "report.html"
    generate_html_report(fixtures, predictions, match_stats_list, value_bets, html_path)
    print(f"HTML report saved to {html_path}")

    # Optional file export
    if args.export:
        from datetime import date
        from src.analysis.export_service import ExportService
        export_service = ExportService()
        exports_dir = Path(config.output.exports_dir)
        exports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = date.today().strftime("%Y-%m-%d")
        if args.export == "csv":
            out_path = exports_dir / f"predictions_{timestamp}.csv"
            export_service.to_csv(predictions, fixtures, value_bets, output_path=out_path)
            print(f"CSV export saved to {out_path}")
        elif args.export == "excel":
            out_path = exports_dir / f"predictions_{timestamp}.xlsx"
            export_service.to_excel(predictions, fixtures, value_bets, output_path=out_path)
            print(f"Excel export saved to {out_path}")

    # Open in browser
    import webbrowser
    webbrowser.open(f"file://{html_path.resolve()}")


if __name__ == "__main__":
    main()
