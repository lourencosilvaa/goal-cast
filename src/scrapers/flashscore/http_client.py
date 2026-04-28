from typing import Optional

import requests

from src.scrapers.flashscore.exceptions import FlashScoreHttpError
from src.scrapers.flashscore.parser import FlashScoreParser


class FlashScoreHttpClient:
    """Fetches FlashScore data via reverse-engineered internal HTTP API."""

    _FIXTURES_PATH = "f_1_{slug}_1_en_1"
    _RESULTS_PATH = "f_2_{slug}_1_en_1"

    def __init__(self, config: object) -> None:
        self.config = config  # type: ignore[assignment]
        self._token: Optional[str] = None

    def get_token(self) -> str:
        if self._token:
            return self._token
        try:
            resp = requests.get(
                self.config.token_url,  # type: ignore[attr-defined]
                headers=self._headers(),
                timeout=self.config.request_timeout,  # type: ignore[attr-defined]
            )
            resp.raise_for_status()
        except Exception as exc:
            raise FlashScoreHttpError(f"Failed to fetch token page: {exc}") from exc

        token = FlashScoreParser.extract_token(resp.text)
        if not token:
            raise FlashScoreHttpError("x-fsign token not found in page JS")

        self._token = token
        return self._token

    def fetch_fixtures(self, league_code: str) -> str:
        return self._fetch(league_code, self._FIXTURES_PATH)

    def fetch_results(self, league_code: str) -> str:
        return self._fetch(league_code, self._RESULTS_PATH)

    def _fetch(self, league_code: str, path_template: str) -> str:
        slug = self._resolve_slug(league_code)
        token = self.get_token()
        path = path_template.format(slug=slug)
        url = f"{self.config.api_url}/{path}"  # type: ignore[attr-defined]

        try:
            resp = requests.get(
                url,
                headers={**self._headers(), "x-fsign": token},
                timeout=self.config.request_timeout,  # type: ignore[attr-defined]
            )
            resp.raise_for_status()
        except Exception as exc:
            raise FlashScoreHttpError(
                f"Feed request failed for {league_code}: {exc}"
            ) from exc

        return resp.text

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
