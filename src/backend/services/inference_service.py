"""On-demand prediction service: delegates inference to the HF Space API."""

from typing import Any, ClassVar

import httpx


class InferenceService:
    """Async HTTP client that calls the HuggingFace Space inference endpoint."""

    #: Seconds allowed for a model call (cold Spaces load the model on demand)
    #: and for the cheap read-only lookups served from memory.
    PREDICTION_TIMEOUT: ClassVar[int] = 120
    LOOKUP_TIMEOUT: ClassVar[int] = 30

    def __init__(self, config: Any) -> None:
        self.config = config

    def _space_url(self) -> str:
        url: str = self.config.inference.space_url.rstrip("/")
        if not url:
            raise RuntimeError(
                "HF Space URL is not configured. Set the HF_SPACE_URL environment variable."
            )
        return url

    async def run(
        self,
        target_date: str | None = None,
        league_codes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.config.inference.enabled:
            raise RuntimeError("On-demand inference is disabled in configuration")

        async with httpx.AsyncClient(timeout=self.PREDICTION_TIMEOUT) as client:
            response = await client.post(
                f"{self._space_url()}/infer",
                json={"date": target_date, "league_codes": league_codes},
            )
        response.raise_for_status()
        data: list[dict[str, Any]] = response.json().get("predictions", [])
        return data

    async def predict_custom(
        self,
        home_team: str,
        away_team: str,
        league_code: str,
    ) -> dict[str, Any]:
        if not self.config.inference.enabled:
            raise RuntimeError("On-demand inference is disabled in configuration")

        async with httpx.AsyncClient(timeout=self.PREDICTION_TIMEOUT) as client:
            response = await client.post(
                f"{self._space_url()}/predict-custom",
                json={
                    "home_team": home_team,
                    "away_team": away_team,
                    "league_code": league_code,
                },
            )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    async def get_teams(self) -> dict[str, list[str]]:
        if not self.config.inference.enabled:
            raise RuntimeError("On-demand inference is disabled in configuration")

        async with httpx.AsyncClient(timeout=self.LOOKUP_TIMEOUT) as client:
            response = await client.get(f"{self._space_url()}/teams")
        response.raise_for_status()
        result: dict[str, list[str]] = response.json()
        return result

    async def get_match_insights(
        self,
        home_team: str,
        away_team: str,
        league_code: str,
    ) -> dict[str, Any]:
        """Head-to-head history and both team profiles for a face-off."""
        if not self.config.inference.enabled:
            raise RuntimeError("On-demand inference is disabled in configuration")

        async with httpx.AsyncClient(timeout=self.LOOKUP_TIMEOUT) as client:
            response = await client.post(
                f"{self._space_url()}/match-insights",
                json={
                    "home_team": home_team,
                    "away_team": away_team,
                    "league_code": league_code,
                },
            )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    async def get_team_insights(self, team: str, league_code: str) -> dict[str, Any]:
        """Full statistical profile of a single team."""
        if not self.config.inference.enabled:
            raise RuntimeError("On-demand inference is disabled in configuration")

        async with httpx.AsyncClient(timeout=self.LOOKUP_TIMEOUT) as client:
            response = await client.get(
                f"{self._space_url()}/team-insights",
                params={"team": team, "league_code": league_code},
            )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result
