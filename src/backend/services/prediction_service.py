"""
Prediction service — orchestrates the ML pipeline with caching.

Key optimisations over the original per-fixture approach:
 1. Historical CSVs are downloaded once and cached on disk (24 h TTL).
 2. The fixtures CSV is downloaded once and cached on disk (1 h TTL).
 3. Feature engineering + ELO are computed ONCE per league, not per fixture.
 4. All fixtures in a league are predicted in a single batch.
 5. Results are held in an in-memory cache (30 min TTL) so subsequent
    requests are instant.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.config_loader import AppConfig
from src.analysis.match_stats import MatchStats, MatchStatsCalculator
from src.analysis.value_detector import ValueBet, ValueDetector
from src.models.data_cleaner import DataCleaner
from src.models.data_loader import FootballDataLoader
from src.models.elo import FootballELO
from src.models.feature_engineer import FeatureEngineer
from src.models.football_data_org_loader import (
    ConfigurationError,
    FootballDataOrgLoader,
    OrgLoaderConfig,
)
from src.models.predictor import MatchPrediction, MatchPredictor
from src.scrapers.base_scraper import ScrapedOdds
from src.scrapers.fixtures_fetcher import (
    Fixture,
    fetch_available_dates,
    fetch_fixtures,
)
from src.scrapers.odds_aggregator import AggregatedOdds

# Cache TTL in seconds (30 minutes)
CACHE_TTL = 30 * 60
# Maximum number of league:date entries kept in memory
MAX_CACHE_ENTRIES = 60


@dataclass
class LeaguePredictions:
    """Cached predictions for a single league."""

    league_code: str
    league_name: str
    fixtures: list[Fixture]
    predictions: list[MatchPrediction]
    match_stats: list[MatchStats | None]
    value_bets: list[ValueBet]
    timestamp: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > CACHE_TTL


class PredictionService:
    """Manages predictions with caching."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._cache: OrderedDict[str, LeaguePredictions] = OrderedDict()
        self._model_dir = Path(config.output.models_dir)
        self._predictor: MatchPredictor | None = None
        self._league_data_cache: dict[str, pd.DataFrame] = {}
        self._org_loader: FootballDataOrgLoader | None = None
        self._org_loader_initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_league_predictions(
        self, league_code: str, target_date: str | None = None
    ) -> LeaguePredictions:
        """Get predictions for a league, using cache if available."""
        resolved_date = target_date or datetime.now().strftime("%d/%m/%Y")
        cache_key = f"{league_code}:{resolved_date}"

        if cache_key in self._cache and not self._cache[cache_key].is_expired():
            self._cache.move_to_end(cache_key)  # LRU touch
            return self._cache[cache_key]

        if self._is_org_league(league_code):
            result = self._compute_predictions_org(league_code, resolved_date)
        else:
            result = self._compute_predictions(league_code, resolved_date)
        self._cache[cache_key] = result
        # Evict oldest entries when cache grows too large
        while len(self._cache) > MAX_CACHE_ENTRIES:
            self._cache.popitem(last=False)
        return result

    def get_all_leagues_predictions(
        self, target_date: str | None = None
    ) -> list[LeaguePredictions]:
        """Get predictions for all supported leagues (domestic + org)."""
        results = []
        for league_code in self.config.data.leagues:
            try:
                result = self.get_league_predictions(league_code, target_date)
                if result.fixtures:
                    results.append(result)
            except Exception as e:
                print(f"Error loading {league_code}: {e}")

        org_loader = self._get_org_loader()
        if org_loader:
            for league_code in self.config.football_data_org.competitions:
                try:
                    result = self.get_league_predictions(league_code, target_date)
                    if result.fixtures:
                        results.append(result)
                except Exception as e:
                    print(f"Error loading org league {league_code}: {e}")

        return results

    def get_available_dates(self) -> list[str]:
        """Return sorted fixture dates (DD/MM/YYYY) from CSV and org API."""
        from datetime import datetime as _dt

        league_codes = list(self.config.data.leagues.keys())
        dates: set[str] = set(fetch_available_dates(league_codes))

        org_loader = self._get_org_loader()
        if org_loader:
            for competition in self.config.football_data_org.competitions:
                try:
                    org_dates = org_loader.fetch_available_upcoming_dates(competition)
                    dates.update(org_dates)
                except Exception as e:
                    print(f"Error fetching org dates for {competition}: {e}")

        return sorted(dates, key=lambda d: _dt.strptime(d, "%d/%m/%Y"))

    def invalidate_cache(self, league_code: str | None = None) -> None:
        """Clear cached predictions."""
        if league_code:
            keys_to_remove = [k for k in self._cache if k.startswith(league_code)]
            for k in keys_to_remove:
                del self._cache[k]
            self._league_data_cache.pop(league_code, None)
        else:
            self._cache.clear()
            self._league_data_cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_org_loader(self) -> FootballDataOrgLoader | None:
        """Lazily create org loader; returns None when API key is not configured."""
        if self._org_loader_initialized:
            return self._org_loader
        self._org_loader_initialized = True
        try:
            org_config = self.config.football_data_org
            loader_config = OrgLoaderConfig(
                api_key=org_config.api_key,
                base_url=org_config.base_url,
                request_timeout=org_config.request_timeout,
                competitions=dict(org_config.competitions),
            )
            self._org_loader = FootballDataOrgLoader(loader_config)
        except ConfigurationError:
            self._org_loader = None
        return self._org_loader

    def _is_org_league(self, league_code: str) -> bool:
        return league_code in self.config.football_data_org.competitions

    def _get_predictor(self) -> MatchPredictor | None:
        """Lazily load the ML predictor (once)."""
        if self._predictor is not None:
            return self._predictor
        model_file = self._model_dir / "ensemble_model.joblib"
        if model_file.exists():
            self._predictor = MatchPredictor(str(self._model_dir))
        return self._predictor

    def _load_league_featured_data(self, league_code: str) -> pd.DataFrame:
        """
        Load historical data for a league, clean it, build features
        and compute ELO — once.  Results are cached in memory.
        """
        if league_code in self._league_data_cache:
            return self._league_data_cache[league_code]

        loader = FootballDataLoader(self.config.data)
        frames = []
        for season in self.config.data.seasons:
            df = loader.load_season(league_code, season)
            if not df.empty:
                frames.append(df)

        if not frames:
            empty = pd.DataFrame()
            self._league_data_cache[league_code] = empty
            return empty

        data = pd.concat(frames, ignore_index=True)

        cleaner = DataCleaner()
        data = cleaner.clean(data)

        engineer = FeatureEngineer(self.config.features)
        data = engineer.build_all_features(data)

        if data.empty:
            self._league_data_cache[league_code] = data
            return data

        elo = FootballELO(self.config.features.elo)
        data = elo.compute_elo_features(data)

        self._league_data_cache[league_code] = data
        return data

    def _predict_fixture_from_data(
        self,
        fixture: Fixture,
        featured_data: pd.DataFrame,
        predictor: MatchPredictor,
    ) -> MatchPrediction:
        """Build a feature row for *one* fixture from pre-computed data."""
        home = fixture.home_team
        away = fixture.away_team

        home_rows = featured_data[
            (featured_data["HomeTeam"] == home) | (featured_data["AwayTeam"] == home)
        ]
        away_rows = featured_data[
            (featured_data["HomeTeam"] == away) | (featured_data["AwayTeam"] == away)
        ]

        if home_rows.empty or away_rows.empty:
            return self._predict_from_odds(fixture)

        last_home_row = home_rows.iloc[-1]
        last_away_row = away_rows.iloc[-1]

        home_was_home = last_home_row.get("HomeTeam") == home
        away_was_away = last_away_row.get("AwayTeam") == away

        def _get_team_val(row, was_expected, feat_expected, feat_opposite):
            val = row.get(feat_expected if was_expected else feat_opposite, 0)
            return val if pd.notna(val) else 0

        feature_row: dict = {}
        for feat in predictor.feature_names:
            if feat.startswith("home_"):
                suffix = feat[len("home_") :]
                feature_row[feat] = _get_team_val(
                    last_home_row,
                    home_was_home,
                    f"home_{suffix}",
                    f"away_{suffix}",
                )
            elif feat.startswith("away_"):
                suffix = feat[len("away_") :]
                feature_row[feat] = _get_team_val(
                    last_away_row,
                    away_was_away,
                    f"away_{suffix}",
                    f"home_{suffix}",
                )
            elif feat.startswith("diff_"):
                suffix = feat[len("diff_") :]
                h = _get_team_val(
                    last_home_row,
                    home_was_home,
                    f"home_{suffix}",
                    f"away_{suffix}",
                )
                a = _get_team_val(
                    last_away_row,
                    away_was_away,
                    f"away_{suffix}",
                    f"home_{suffix}",
                )
                feature_row[feat] = h - a
            elif feat == "elo_home":
                feature_row[feat] = _get_team_val(
                    last_home_row,
                    home_was_home,
                    "elo_home",
                    "elo_away",
                )
            elif feat == "elo_away":
                feature_row[feat] = _get_team_val(
                    last_away_row,
                    away_was_away,
                    "elo_away",
                    "elo_home",
                )
            elif feat == "elo_diff":
                h_elo = _get_team_val(
                    last_home_row,
                    home_was_home,
                    "elo_home",
                    "elo_away",
                )
                a_elo = _get_team_val(
                    last_away_row,
                    away_was_away,
                    "elo_away",
                    "elo_home",
                )
                feature_row[feat] = h_elo - a_elo
            elif feat == "elo_expected_home":
                feature_row[feat] = _get_team_val(
                    last_home_row,
                    home_was_home,
                    "elo_expected_home",
                    "elo_expected_away",
                )
            elif feat == "elo_expected_away":
                feature_row[feat] = _get_team_val(
                    last_away_row,
                    away_was_away,
                    "elo_expected_away",
                    "elo_expected_home",
                )
            elif feat == "xG_diff":
                h_xg = _get_team_val(
                    last_home_row,
                    home_was_home,
                    "home_xG_rolling",
                    "away_xG_rolling",
                )
                a_xg = _get_team_val(
                    last_away_row,
                    away_was_away,
                    "away_xG_rolling",
                    "home_xG_rolling",
                )
                feature_row[feat] = h_xg - a_xg
            elif feat == "rest_advantage":
                h_rest = _get_team_val(
                    last_home_row,
                    home_was_home,
                    "home_rest_days",
                    "away_rest_days",
                )
                a_rest = _get_team_val(
                    last_away_row,
                    away_was_away,
                    "away_rest_days",
                    "home_rest_days",
                )
                feature_row[feat] = h_rest - a_rest
            elif feat == "avg_draw_pct":
                h_dp = _get_team_val(
                    last_home_row,
                    home_was_home,
                    "home_draw_pct",
                    "away_draw_pct",
                )
                a_dp = _get_team_val(
                    last_away_row,
                    away_was_away,
                    "away_draw_pct",
                    "home_draw_pct",
                )
                feature_row[feat] = (h_dp + a_dp) / 2
            elif feat == "form_gap":
                h_f = _get_team_val(
                    last_home_row,
                    home_was_home,
                    "home_Form",
                    "away_Form",
                )
                a_f = _get_team_val(
                    last_away_row,
                    away_was_away,
                    "away_Form",
                    "home_Form",
                )
                feature_row[feat] = abs(h_f - a_f)
            elif feat == "attack_similarity":
                h_gf = _get_team_val(
                    last_home_row,
                    home_was_home,
                    "home_avg_GF",
                    "away_avg_GF",
                )
                a_gf = _get_team_val(
                    last_away_row,
                    away_was_away,
                    "away_avg_GF",
                    "home_avg_GF",
                )
                feature_row[feat] = 1 / (1 + abs(h_gf - a_gf))
            elif feat == "defense_similarity":
                h_ga = _get_team_val(
                    last_home_row,
                    home_was_home,
                    "home_avg_GA",
                    "away_avg_GA",
                )
                a_ga = _get_team_val(
                    last_away_row,
                    away_was_away,
                    "away_avg_GA",
                    "home_avg_GA",
                )
                feature_row[feat] = 1 / (1 + abs(h_ga - a_ga))
            elif feat == "combined_defensive":
                h_ga = _get_team_val(
                    last_home_row,
                    home_was_home,
                    "home_avg_GA",
                    "away_avg_GA",
                )
                a_ga = _get_team_val(
                    last_away_row,
                    away_was_away,
                    "away_avg_GA",
                    "home_avg_GA",
                )
                feature_row[feat] = 1 / (1 + h_ga) + 1 / (1 + a_ga)
            elif feat == "is_midweek":
                feature_row[feat] = 0
            elif feat.startswith("h2h_"):
                h2h_rows = featured_data[
                    (
                        (featured_data["HomeTeam"] == home)
                        & (featured_data["AwayTeam"] == away)
                    )
                    | (
                        (featured_data["HomeTeam"] == away)
                        & (featured_data["AwayTeam"] == home)
                    )
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
        return predictions[0] if predictions else self._predict_from_odds(fixture)

    @staticmethod
    def _predict_from_odds(fixture: Fixture) -> MatchPrediction:
        """Fallback: derive prediction from B365 implied probabilities.
        When no odds are available, uses a slight home-advantage prior."""
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

    @staticmethod
    def _fixture_to_odds(fixture: Fixture) -> AggregatedOdds:
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

    def _compute_predictions_org(
        self, league_code: str, target_date: str
    ) -> LeaguePredictions:
        """Prediction pipeline for football-data.org competitions (no CSV history)."""
        league_name = self.config.football_data_org.competitions.get(
            league_code, league_code
        )
        org_loader = self._get_org_loader()
        if not org_loader:
            return LeaguePredictions(
                league_code=league_code,
                league_name=league_name,
                fixtures=[],
                predictions=[],
                match_stats=[],
                value_bets=[],
            )

        raw_fixtures = org_loader.fetch_upcoming_fixtures(league_code, target_date)
        fixtures = [
            Fixture(
                division=f["division"],
                league=f["league"],
                date=f["date"],
                time=f["time"],
                home_team=f["home_team"],
                away_team=f["away_team"],
                b365_home=0.0,
                b365_draw=0.0,
                b365_away=0.0,
            )
            for f in raw_fixtures
        ]

        if not fixtures:
            return LeaguePredictions(
                league_code=league_code,
                league_name=league_name,
                fixtures=[],
                predictions=[],
                match_stats=[],
                value_bets=[],
            )

        predictions = [self._predict_from_odds(f) for f in fixtures]
        return LeaguePredictions(
            league_code=league_code,
            league_name=league_name,
            fixtures=fixtures,
            predictions=predictions,
            match_stats=[None] * len(fixtures),
            value_bets=[],
        )

    def _compute_predictions(
        self, league_code: str, target_date: str | None
    ) -> LeaguePredictions:
        """Run the full prediction pipeline for a league — in batch."""
        league_name = self.config.data.leagues.get(league_code, league_code)

        # 1. Fetch fixtures (cached on disk for 1 h)
        fixtures = fetch_fixtures(target_date=target_date, leagues=[league_code])

        if not fixtures:
            return LeaguePredictions(
                league_code=league_code,
                league_name=league_name,
                fixtures=[],
                predictions=[],
                match_stats=[],
                value_bets=[],
            )

        # 2. Load historical data ONCE for the league
        predictor = self._get_predictor()
        featured_data = (
            self._load_league_featured_data(league_code)
            if predictor
            else pd.DataFrame()
        )

        # 3. Predict every fixture using the shared featured_data
        predictions: list[MatchPrediction] = []
        odds_list: list[AggregatedOdds] = []
        for fixture in fixtures:
            if predictor and not featured_data.empty:
                pred = self._predict_fixture_from_data(
                    fixture, featured_data, predictor
                )
            else:
                pred = self._predict_from_odds(fixture)
            predictions.append(pred)
            odds_list.append(self._fixture_to_odds(fixture))

        # 4. Compute Poisson match stats (reuses its own internal cache)
        stats_calc = MatchStatsCalculator(self.config.data)
        match_stats: list[MatchStats | None] = []
        for fixture in fixtures:
            try:
                stats = stats_calc.compute_match_stats(
                    fixture.division, fixture.home_team, fixture.away_team
                )
                match_stats.append(stats)
            except Exception:
                match_stats.append(None)

        # 5. Detect value bets
        detector = ValueDetector(self.config.analysis)
        value_bets = detector.find_value_bets_batch(predictions, odds_list)

        return LeaguePredictions(
            league_code=league_code,
            league_name=league_name,
            fixtures=fixtures,
            predictions=predictions,
            match_stats=match_stats,
            value_bets=value_bets,
        )
