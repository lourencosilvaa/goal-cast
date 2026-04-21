import time
from dataclasses import dataclass

from config.config_loader import ScrapersConfig
from src.scrapers.base_scraper import BaseScraper, ScrapedOdds
from src.scrapers.betano_scraper import BetanoScraper
from src.scrapers.betclic_scraper import BetclicScraper
from src.scrapers.solverde_scraper import SolverdeScraper


@dataclass
class AggregatedOdds:
    """Aggregated odds from multiple betting sources for one match."""

    home_team: str
    away_team: str
    league: str
    match_date: str
    sources: list[ScrapedOdds]

    @property
    def avg_home_win(self) -> float:
        """Average home win odds across all sources."""
        values = [s.home_win for s in self.sources if s.home_win > 0]
        return sum(values) / len(values) if values else 0

    @property
    def avg_draw(self) -> float:
        """Average draw odds across all sources."""
        values = [s.draw for s in self.sources if s.draw > 0]
        return sum(values) / len(values) if values else 0

    @property
    def avg_away_win(self) -> float:
        """Average away win odds across all sources."""
        values = [s.away_win for s in self.sources if s.away_win > 0]
        return sum(values) / len(values) if values else 0

    @property
    def best_home_win(self) -> float:
        """Best (highest) home win odds across sources."""
        values = [s.home_win for s in self.sources if s.home_win > 0]
        return max(values) if values else 0

    @property
    def best_draw(self) -> float:
        """Best (highest) draw odds across sources."""
        values = [s.draw for s in self.sources if s.draw > 0]
        return max(values) if values else 0

    @property
    def best_away_win(self) -> float:
        """Best (highest) away win odds across sources."""
        values = [s.away_win for s in self.sources if s.away_win > 0]
        return max(values) if values else 0

    def avg_implied_probabilities(self) -> dict[str, float]:
        """Average implied probabilities (normalized) across sources."""
        all_probs = [s.implied_probabilities() for s in self.sources]
        if not all_probs:
            return {"home": 0, "draw": 0, "away": 0}

        avg = {
            "home": sum(p["home"] for p in all_probs) / len(all_probs),
            "draw": sum(p["draw"] for p in all_probs) / len(all_probs),
            "away": sum(p["away"] for p in all_probs) / len(all_probs),
        }
        total = sum(avg.values())
        if total > 0:
            return {k: v / total for k, v in avg.items()}
        return avg

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "league": self.league,
            "match_date": self.match_date,
            "sources_count": len(self.sources),
            "average_odds": {
                "home_win": round(self.avg_home_win, 2),
                "draw": round(self.avg_draw, 2),
                "away_win": round(self.avg_away_win, 2),
            },
            "best_odds": {
                "home_win": round(self.best_home_win, 2),
                "draw": round(self.best_draw, 2),
                "away_win": round(self.best_away_win, 2),
            },
            "implied_probabilities": {
                k: round(v, 4) for k, v in self.avg_implied_probabilities().items()
            },
            "per_source": [s.to_dict() for s in self.sources],
        }


class OddsAggregator:
    """Aggregates odds from multiple Portuguese betting sites."""

    def __init__(self, config: ScrapersConfig) -> None:
        self.config = config
        self.scrapers: list[BaseScraper] = []
        self._init_scrapers()

    def _init_scrapers(self) -> None:
        """Initialize enabled scrapers from config."""
        if self.config.betclic.enabled:
            self.scrapers.append(
                BetclicScraper(
                    base_url=self.config.betclic.base_url,
                    user_agent=self.config.user_agent,
                    timeout=self.config.request_timeout,
                )
            )
        if self.config.betano.enabled:
            self.scrapers.append(
                BetanoScraper(
                    base_url=self.config.betano.base_url,
                    user_agent=self.config.user_agent,
                    timeout=self.config.request_timeout,
                )
            )
        if self.config.solverde.enabled:
            self.scrapers.append(
                SolverdeScraper(
                    base_url=self.config.solverde.base_url,
                    user_agent=self.config.user_agent,
                    timeout=self.config.request_timeout,
                )
            )

    def scrape_all_leagues(self, league_name: str) -> list[AggregatedOdds]:
        """Scrape a specific league across all enabled sources and aggregate."""
        all_odds: dict[str, list[ScrapedOdds]] = {}

        for scraper in self.scrapers:
            leagues = scraper.get_available_leagues()
            league_url = leagues.get(league_name)

            if not league_url:
                print(f"  [{scraper.source_name}] " f"League '{league_name}' not found")
                continue

            print(f"  Scraping {scraper.source_name} - {league_name}...")
            matches = scraper.scrape_league(league_url)
            print(f"    Found {len(matches)} matches")

            for match in matches:
                key = self._normalize_match_key(match.home_team, match.away_team)
                if key not in all_odds:
                    all_odds[key] = []
                all_odds[key].append(match)

            time.sleep(self.config.rate_limit_seconds)

        return self._aggregate_odds(all_odds)

    def scrape_today(self) -> list[AggregatedOdds]:
        """Scrape all today's matches across all enabled sources."""
        all_odds: dict[str, list[ScrapedOdds]] = {}

        for scraper in self.scrapers:
            leagues = scraper.get_available_leagues()

            for league_name, league_url in leagues.items():
                print(f"  Scraping {scraper.source_name} - {league_name}...")
                matches = scraper.scrape_league(league_url)
                print(f"    Found {len(matches)} matches")

                for match in matches:
                    key = self._normalize_match_key(match.home_team, match.away_team)
                    if key not in all_odds:
                        all_odds[key] = []
                    all_odds[key].append(match)

                time.sleep(self.config.rate_limit_seconds)

        return self._aggregate_odds(all_odds)

    def _aggregate_odds(
        self, all_odds: dict[str, list[ScrapedOdds]]
    ) -> list[AggregatedOdds]:
        """Group odds by match and create aggregated entries."""
        result: list[AggregatedOdds] = []

        for _, odds_list in all_odds.items():
            if not odds_list:
                continue

            first = odds_list[0]
            result.append(
                AggregatedOdds(
                    home_team=first.home_team,
                    away_team=first.away_team,
                    league=first.league,
                    match_date=first.match_date,
                    sources=odds_list,
                )
            )

        return result

    @staticmethod
    def _normalize_match_key(home: str, away: str) -> str:
        """Create a normalized key for matching across sources."""
        h = home.lower().strip()
        a = away.lower().strip()
        return f"{h}_vs_{a}"
