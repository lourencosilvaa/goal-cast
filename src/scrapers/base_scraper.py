from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ScrapedOdds:
    """Normalized odds scraped from a betting site."""

    source: str
    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    league: str
    match_date: str
    url: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "odds": {
                "home_win": round(self.home_win, 2),
                "draw": round(self.draw, 2),
                "away_win": round(self.away_win, 2),
            },
            "implied_probabilities": {
                "home_win": round(1 / self.home_win, 4) if self.home_win > 0 else 0,
                "draw": round(1 / self.draw, 4) if self.draw > 0 else 0,
                "away_win": round(1 / self.away_win, 4) if self.away_win > 0 else 0,
            },
            "league": self.league,
            "match_date": self.match_date,
            "url": self.url,
        }

    def implied_probabilities(self) -> dict[str, float]:
        """Convert odds to implied probabilities (normalized)."""
        raw_h = 1 / self.home_win if self.home_win > 0 else 0
        raw_d = 1 / self.draw if self.draw > 0 else 0
        raw_a = 1 / self.away_win if self.away_win > 0 else 0
        total = raw_h + raw_d + raw_a

        if total == 0:
            return {"home": 0, "draw": 0, "away": 0}

        return {
            "home": raw_h / total,
            "draw": raw_d / total,
            "away": raw_a / total,
        }


class BaseScraper(ABC):
    """Abstract base class for betting site scrapers."""

    def __init__(self, base_url: str, user_agent: str, timeout: int) -> None:
        self.base_url = base_url
        self.user_agent = user_agent
        self.timeout = timeout

    @abstractmethod
    def scrape_league(self, league_url: str) -> list[ScrapedOdds]:
        """Scrape all available matches for a given league page."""
        ...

    @abstractmethod
    def scrape_match(self, match_url: str) -> ScrapedOdds | None:
        """Scrape odds for a specific match."""
        ...

    @abstractmethod
    def get_available_leagues(self) -> dict[str, str]:
        """Return available leagues and their URLs."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """The name of this betting source."""
        ...
