import time

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from src.scrapers.base_scraper import BaseScraper, ScrapedOdds


class BetanoScraper(BaseScraper):
    """Scraper for Betano.pt football odds."""

    @property
    def source_name(self) -> str:
        return "Betano"

    def get_available_leagues(self) -> dict[str, str]:
        """Return available football leagues on Betano."""
        return {
            "Liga Portugal": f"{self.base_url}portugal/liga-portugal/",
            "Premier League": f"{self.base_url}inglaterra/premier-league/",
            "La Liga": f"{self.base_url}espanha/laliga/",
            "Serie A": f"{self.base_url}italia/serie-a/",
            "Bundesliga": f"{self.base_url}alemanha/bundesliga/",
            "Ligue 1": f"{self.base_url}franca/ligue-1/",
            "Champions League": f"{self.base_url}europa/liga-dos-campeoes-da-uefa/",
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

                time.sleep(3)

                html = page.content()
                soup = BeautifulSoup(html, "html.parser")

                event_rows = soup.select(
                    "[class*='events-list__grid'], "
                    "[class*='event-row'], "
                    "[class*='vue-recycle-scroller__item-view']"
                )

                for row in event_rows:
                    parsed = self._parse_event_row(row, league_url)
                    if parsed:
                        matches.append(parsed)

                browser.close()

        except Exception as e:
            print(f"  [Betano] Error scraping {league_url}: {e}")

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
                time.sleep(3)

                html = page.content()
                soup = BeautifulSoup(html, "html.parser")

                result = self._parse_match_page(soup, match_url)
                browser.close()
                return result

        except Exception as e:
            print(f"  [Betano] Error scraping match {match_url}: {e}")
            return None

    def _parse_event_row(
        self, row: BeautifulSoup, league_url: str  # type: ignore[override]
    ) -> ScrapedOdds | None:
        """Parse a single event row from the league page."""
        try:
            participants = row.select("[class*='participant'], [class*='team-name']")
            odds_buttons = row.select(
                "[class*='selection__odd'], "
                "[class*='odd-button'], "
                "[class*='selection-price']"
            )

            if len(participants) < 2 or len(odds_buttons) < 3:
                return None

            home_team = participants[0].get_text(strip=True)
            away_team = participants[1].get_text(strip=True)

            odds_values = []
            for btn in odds_buttons[:3]:
                text = btn.get_text(strip=True).replace(",", ".")
                try:
                    odds_values.append(float(text))
                except ValueError:
                    return None

            if len(odds_values) < 3 or any(v <= 1.0 for v in odds_values):
                return None

            date_elem = row.select_one("[class*='datetime'], [class*='date']")
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
            print(f"  [Betano] Parse error: {e}")
            return None

    def _parse_match_page(self, soup: BeautifulSoup, url: str) -> ScrapedOdds | None:
        """Parse a specific match page."""
        try:
            participants = soup.select("[class*='participant'], [class*='team-name']")
            odds_buttons = soup.select(
                "[class*='selection__odd'], "
                "[class*='odd-button'], "
                "[class*='selection-price']"
            )

            if len(participants) < 2 or len(odds_buttons) < 3:
                return None

            home_team = participants[0].get_text(strip=True)
            away_team = participants[1].get_text(strip=True)

            odds_values = []
            for btn in odds_buttons[:3]:
                text = btn.get_text(strip=True).replace(",", ".")
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
        if len(parts) >= 2:
            return parts[-1].replace("-", " ").title()
        return ""
