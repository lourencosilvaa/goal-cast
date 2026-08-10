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
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from src.analysis.match_stats import MatchStatsCalculator
from src.analysis.value_detector import ValueDetector
from src.models.data_cleaner import DataCleaner
from src.models.data_loader import FootballDataLoader
from src.corpus.european_corpus import load_european_corpus
from src.models.cross_competition import (
    CrossCompetitionCorpus,
    CrossCompetitionEloBuilder,
)
from src.models.elo import FootballELO
from src.models.cross_league import match_counts
from src.models.feature_engineer import FeatureEngineer
from src.models.fixture_features import FixtureFeatureBuilder, FixtureTeams
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


def _load_all_featured_data(config) -> tuple[pd.DataFrame, Any]:
    """Every tracked league, feature-engineered, with ELO walked once.

    The ELO walk must span **every** league, exactly as training does, and only
    then may the result be narrowed to one league. Walking a single league in
    isolation gives a different answer, and the difference is not small:
    measured on real data, Arsenal came out 27.8 points lower.

    That equivalence used to hold. Domestic leagues are disjoint pools, so a
    per-league walk was bit-identical to a combined one — the tests still pin
    that. The European corpus deliberately links the pools, which is the point
    of the calibration, and the same linkage is what breaks the shortcut: a
    team's European opponent carries a rating built from *its* domestic
    history, so a walk missing that league starts the opponent at the default
    and mis-scores the tie for both sides.

    Loaded once for the whole run rather than per league, since every league
    now needs all the others anyway.

    Returns the featured frame **and the fitted ELO instance**. The instance is
    the only place the calibrated ratings survive: ``build`` returns domestic
    rows, so a European result updates the ratings without appearing in the
    output, and recomputing ELO from that output would silently discard every
    one of them.
    """
    loader = FootballDataLoader(config.data, hf_config=config.huggingface)
    elo = FootballELO(config.features.elo)
    frames = []
    for league_code in config.data.leagues:
        for season in config.data.seasons:
            df = loader.load_season(league_code, season)
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame(), elo

    data = pd.concat(frames, ignore_index=True)
    del frames

    cleaner = DataCleaner()
    data = cleaner.clean(data)

    engineer = FeatureEngineer(config.features)
    data = engineer.build_all_features(data)

    if data.empty:
        return data, elo

    # Same corpus, same builder and now the same domestic pool as training. A
    # fixture's elo_home is read from that team's last historical row, so
    # reproducing it means replaying the identical chronological walk.
    european = load_european_corpus(config, verbose=False)
    built = CrossCompetitionEloBuilder(elo).build(
        CrossCompetitionCorpus(domestic=data, supplementary=european)
    )
    return built, elo


def _for_league(featured_data: pd.DataFrame, league_code: str) -> pd.DataFrame:
    """One league's rows, taken *after* the cross-league ELO walk."""
    if featured_data.empty:
        return featured_data
    column = "League" if "League" in featured_data.columns else "Div"
    return featured_data[featured_data[column] == league_code]


def _match_counts(featured_data: pd.DataFrame) -> dict[str, int]:
    """How many matches each team appears in, home or away.

    Delegates to :func:`src.models.cross_league.match_counts`, which the Space
    needs too — the copy that would otherwise have been made there is the same
    duplication that produced the feature-row bug.
    """
    return match_counts(featured_data)


def _run_european(
    config, poisson_model, all_featured_data: pd.DataFrame, elo: Any
) -> int:
    """Discover and predict UEFA fixtures. Returns prediction sets uploaded.

    Predicted by Dixon-Coles and ELO rather than the ensemble, and labelled as
    such — see :mod:`src.models.european_predictor` for why the ensemble is
    not valid here.
    """
    european = getattr(config, "european", None)
    if european is None or not european.enabled or not european.prediction.enabled:
        return 0
    if poisson_model is None or all_featured_data.empty:
        print("\nEuropean fixtures skipped: no Poisson model or no history loaded.")
        return 0

    from src.models.european_predictor import EuropeanMatchPredictor
    from src.scrapers.european_fixtures_fetcher import (
        EuropeanFixturesFetcher,
        build_provider_chain,
    )
    from src.teams.registry import load_team_registry
    from src.teams.resolver import StaticTeamAliasRepository, TeamNameResolver

    print(f"\n{'='*60}")
    print("European competitions")

    resolver = TeamNameResolver(
        load_team_registry(config.teams.historical_registry_path),
        StaticTeamAliasRepository(config.teams.aliases.seed_path),
        config.teams.aliases,
    )
    chain = build_provider_chain(european)
    fixtures = EuropeanFixturesFetcher(european, chain, resolver).fetch()
    if not fixtures:
        print("  No upcoming European fixtures found.")
        return 0

    # The ELO instance from the cross-league walk, passed in rather than
    # rebuilt. Recomputing it from `all_featured_data` would look equivalent
    # and silently revert every rating to its uncalibrated value, because that
    # frame holds domestic rows only — the European results that link the
    # leagues update the ratings without ever appearing in it.
    predictor = EuropeanMatchPredictor(
        dixon_coles=poisson_model,
        elo=elo,
        config=european.prediction,
        match_counts=_match_counts(all_featured_data),
    )

    by_competition_date: dict[tuple[str, str], list] = {}
    for resolved in fixtures:
        if not resolved.is_resolved:
            print(
                f"  SKIP {resolved.fixture.home_team} vs "
                f"{resolved.fixture.away_team}: unresolved "
                f"({', '.join(resolved.unresolved_names)})"
            )
            continue
        reason = predictor.refusal_reason(resolved.home_team, resolved.away_team)
        if reason is not None:
            print(f"  SKIP {resolved.home_team} vs {resolved.away_team}: {reason}")
            continue
        prediction = predictor.predict(resolved.home_team, resolved.away_team)
        match_date = resolved.fixture.kickoff.strftime("%d/%m/%Y")
        key = (resolved.fixture.competition, match_date)
        by_competition_date.setdefault(key, []).append(prediction)
        print(
            f"  {resolved.home_team} vs {resolved.away_team}: "
            f"{prediction.predicted_outcome} ({prediction.confidence:.1%})"
        )

    uploaded = 0
    for (competition, match_date), predictions in by_competition_date.items():
        payload = {
            "league_code": competition,
            "league_name": european.competition_names.get(competition, competition),
            "match_date": match_date,
            # Read off the predictor, not a constant: the label follows the
            # configured weight, and a payload naming a model that did not
            # produce these numbers is the thing the field exists to prevent.
            "model": predictor.model_name,
            "fixtures": [p.to_dict() for p in predictions],
            "generated_at": datetime.now().isoformat(),
        }
        _upload_to_supabase(competition, match_date, payload)
        uploaded += 1
    return uploaded


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
    """Predict one unplayed fixture from each side's most recent match.

    The feature row is built by the shared
    :class:`~src.models.fixture_features.FixtureFeatureBuilder`, which the
    HuggingFace Space uses too. It used to be built here, inline, and the Space
    had its own version that copied every pair-dependent feature from the home
    team's previous game — so the two deployments disagreed about the same
    match. One implementation is the fix; see that module for the detail.
    """
    if featured_data.empty:
        return _predict_from_odds(fixture)

    builder = FixtureFeatureBuilder(featured_data)
    feature_row = builder.build(
        FixtureTeams(home=fixture.home_team, away=fixture.away_team),
        predictor.feature_names,
    )

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
            vb
            for vb in value_bets
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
    parser = argparse.ArgumentParser(
        description="Run ML inference and upload to Supabase"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Target date DD/MM/YYYY (default: all fixture dates)",
    )
    parser.add_argument(
        "--leagues",
        default=None,
        help="Comma-separated league codes (default: config data.served_leagues)",
    )
    args = parser.parse_args()

    config = load_config()

    # Resolve leagues.
    #
    # `served_leagues`, not `data.leagues`: the corpus is wider than the
    # product on purpose, and predicting a division nobody can select would
    # only fill Supabase with rows no client ever asks for. The ELO/training
    # load in `_load_all_featured_data` still spans `data.leagues` — narrowing
    # *that* is what moves ratings, and it is deliberately left alone.
    #
    # An explicit --leagues still overrides, including with unserved codes: an
    # operator backfilling one division is not the product surface.
    league_codes = (
        [lc.strip().upper() for lc in args.leagues.split(",")]
        if args.leagues
        else list(config.data.served_leagues)
    )

    # Resolve dates
    if args.date:
        target_dates = [args.date]
    else:
        target_dates = fetch_available_dates(league_codes)
        today = datetime.now().strftime("%d/%m/%Y")
        if today not in target_dates:
            target_dates.append(today)
        target_dates = sorted(
            target_dates, key=lambda d: datetime.strptime(d, "%d/%m/%Y")
        )

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

    # Loaded once for the whole run. ELO has to be walked across every league
    # to match training, so there is nothing left to gain from loading them
    # one at a time — and doing so would repeat the full walk per league.
    all_featured_data = pd.DataFrame()
    fitted_elo = FootballELO(config.features.elo)
    if predictor:
        print("\nLoading historical data for all leagues (ELO spans them all)…")
        all_featured_data, fitted_elo = _load_all_featured_data(config)
        print(f"Historical data: {len(all_featured_data)} rows")

    for league_code in league_codes:
        league_name = config.data.leagues.get(league_code, league_code)
        print(f"\n{'='*60}")
        print(f"League: {league_name} ({league_code})")

        featured_data = _for_league(all_featured_data, league_code)
        if predictor:
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
                odds_list.append(
                    AggregatedOdds(
                        home_team=fixture.home_team,
                        away_team=fixture.away_team,
                        league=fixture.league,
                        match_date=fixture.date,
                        sources=[scraped],
                    )
                )

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

        gc.collect()

    total_uploaded += _run_european(
        config, poisson_model, all_featured_data, fitted_elo
    )

    del all_featured_data
    gc.collect()

    print(f"\nDone. Uploaded {total_uploaded} prediction sets to Supabase.")


if __name__ == "__main__":
    main()
