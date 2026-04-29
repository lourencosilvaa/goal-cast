from src.scrapers.flashscore.exceptions import FlashScoreHttpError


class FlashScoreHttpClient:
    """HTTP client for FlashScore feed API.

    The token-based authentication approach is no longer functional —
    FlashScore moved the feed_sign token into a runtime JS config object
    that cannot be extracted from a static HTTP response. Calls to
    fetch_fixtures / fetch_results will always raise FlashScoreHttpError
    so the scraper falls through to the Playwright client.
    """

    def __init__(self, config: object) -> None:
        self.config = config  # type: ignore[assignment]

    def fetch_fixtures(self, league_code: str) -> str:
        self._resolve_slug(league_code)
        raise FlashScoreHttpError(
            "HTTP token-based fetch is no longer supported; "
            "FlashScore moved the feed_sign token to a runtime JS config. "
            "Use the Playwright client instead."
        )

    def fetch_results(self, league_code: str) -> str:
        self._resolve_slug(league_code)
        raise FlashScoreHttpError(
            "HTTP token-based fetch is no longer supported; "
            "FlashScore moved the feed_sign token to a runtime JS config. "
            "Use the Playwright client instead."
        )

    def _resolve_slug(self, league_code: str) -> str:
        leagues: dict[str, str] = self.config.leagues  # type: ignore[attr-defined]
        if league_code not in leagues:
            raise FlashScoreHttpError(f"Unknown league code: {league_code!r}")
        return leagues[league_code]

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.config.user_agent,  # type: ignore[attr-defined]
            "Accept": "*/*",
            "Referer": self.config.base_url,  # type: ignore[attr-defined]
        }
