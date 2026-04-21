import time

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from src.scrapers.base_scraper import BaseScraper, ScrapedOdds


class BetclicScraper(BaseScraper):
    """Scraper for Betclic.pt football odds."""

    @property
    def source_name(self) -> str:
        return "Betclic"

    def get_available_leagues(self) -> dict[str, str]:
        """Return available football leagues on Betclic."""
        return {
            "Liga Portugal": f"{self.base_url}/liga-portugal-c33",
            "Premier League": f"{self.base_url}/premier-league-c1",
            "La Liga": f"{self.base_url}/la-liga-c7",
            "Serie A": f"{self.base_url}/serie-a-c4",
            "Bundesliga": f"{self.base_url}/bundesliga-c5",
            "Ligue 1": f"{self.base_url}/ligue-1-c6",
            "Champions League": f"{self.base_url}/liga-dos-campeoes-c49",
        }

    def scrape_league(self, league_url: str) -> list[ScrapedOdds]:
        """Scrape all matches for a league page using Playwright."""
        matches: list[ScrapedOdds] = []

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=self.user_agent,
                    viewport={"width": 1920, "height": 1080},
                )
                page = context.new_page()
                page.goto(league_url, timeout=self.timeout * 1000)
                page.wait_for_load_state("networkidle", timeout=self.timeout * 1000)

                time.sleep(2)

                html = page.content()
                soup = BeautifulSoup(html, "html.parser")

                event_cards = soup.select(
                    "[class*='event'], [class*='match'], [class*='sports-event']"
                )

                for card in event_cards:
                    parsed = self._parse_event_card(card, league_url)
                    if parsed:
                        matches.append(parsed)

                browser.close()

        except Exception as e:
            print(f"  [Betclic] Error scraping {league_url}: {e}")

        return matches

    def scrape_match(self, match_url: str) -> ScrapedOdds | None:
        """Scrape odds for a specific match page."""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent=self.user_agent)
                page = context.new_page()
                page.goto(match_url, timeout=self.timeout * 1000)
                page.wait_for_load_state("networkidle", timeout=self.timeout * 1000)
                time.sleep(2)

                html = page.content()
                soup = BeautifulSoup(html, "html.parser")

                result = self._parse_match_page(soup, match_url)
                browser.close()
                return result

        except Exception as e:
            print(f"  [Betclic] Error scraping match {match_url}: {e}")
            return None

    def _parse_event_card(
        self, card: BeautifulSoup, league_url: str  # type: ignore[override]
    ) -> ScrapedOdds | None:
        """Parse a single event card from the league page."""
        try:
            teams = card.select("[class*='team'], [class*='contestant']")
            odds_elements = card.select(
                "[class*='odd'], [class*='price'], [class*='selection']"
            )

            if len(teams) < 2 or len(odds_elements) < 3:
                return None

            home_team = teams[0].get_text(strip=True)
            away_team = teams[1].get_text(strip=True)

            odds_values = []
            for elem in odds_elements[:3]:
                text = elem.get_text(strip=True).replace(",", ".")
                try:
                    odds_values.append(float(text))
                except ValueError:
                    return None

            if len(odds_values) < 3 or any(v <= 1.0 for v in odds_values):
                return None

            date_elem = card.select_one("[class*='date'], [class*='time']")
            match_date = date_elem.get_text(strip=True) if date_elem else ""

            return ScrapedOdds(
                source=self.source_name,
                home_team=home_team,
                away_team=away_team,
                home_win=odds_values[0],
                draw=odds_values[1],
                away_win=odds_values[2],
                league=self._extract_league_name(league_url),
                match_date=match_date,
                url=league_url,
            )

        except (IndexError, ValueError) as e:
            print(f"  [Betclic] Parse error: {e}")
            return None

    def _parse_match_page(self, soup: BeautifulSoup, url: str) -> ScrapedOdds | None:
        """Parse a specific match page."""
        try:
            teams = soup.select("[class*='team'], [class*='contestant']")
            odds_elements = soup.select(
                "[class*='odd'], [class*='price'], [class*='selection']"
            )

            if len(teams) < 2 or len(odds_elements) < 3:
                return None

            home_team = teams[0].get_text(strip=True)
            away_team = teams[1].get_text(strip=True)

            odds_values = []
            for elem in odds_elements[:3]:
                text = elem.get_text(strip=True).replace(",", ".")
                try:
                    odds_values.append(float(text))
                except ValueError:
                    continue

            if len(odds_values) < 3:
                return None

            return ScrapedOdds(
                source=self.source_name,
                home_team=home_team,
                away_team=away_team,
                home_win=odds_values[0],
                draw=odds_values[1],
                away_win=odds_values[2],
                league="",
                match_date="",
                url=url,
            )

        except Exception:
            return None

    @staticmethod
    def _extract_league_name(url: str) -> str:
        """Extract league name from URL."""
        parts = url.rstrip("/").split("/")
        if parts:
            return parts[-1].split("-c")[0].replace("-", " ").title()
        return ""
