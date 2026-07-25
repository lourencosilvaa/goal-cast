from dataclasses import dataclass

from src.scrapers.base_scraper import ScrapedOdds


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
