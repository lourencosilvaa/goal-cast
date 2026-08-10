"""On-demand prediction service: delegates inference to the HF Space API."""

from typing import Any, ClassVar

import httpx


class PredictionRefused(RuntimeError):
    """The Space declined to predict, and said why.

    A refusal is not an outage, and collapsing the two would send the user to
    retry something that will never succeed. It happens when a club has no
    history to predict from — its league is not tracked, or it has too few
    recorded matches for its parameters to mean anything — and the reason is
    the useful part, so it is carried rather than replaced by a status line.

    A ``RuntimeError`` so that a caller which only knows about the old
    behaviour still catches it; callers that care must test for this first.
    """


class InferenceService:
    """Async HTTP client that calls the HuggingFace Space inference endpoint."""

    #: Seconds allowed for a model call (cold Spaces load the model on demand)
    #: and for the cheap read-only lookups served from memory.
    PREDICTION_TIMEOUT: ClassVar[int] = 120
    LOOKUP_TIMEOUT: ClassVar[int] = 30

    #: The status the Space answers a stated refusal with.
    REFUSED_STATUS: ClassVar[int] = 422
    #: Shown when the Space refuses without a body we can read. Rare, and still
    #: better than reporting the service as down.
    UNSTATED_REFUSAL: ClassVar[str] = (
        "The model declined to predict this fixture and gave no reason."
    )

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
        away_league_code: str | None = None,
    ) -> dict[str, Any]:
        """One fixture, priced by whichever model the pairing calls for.

        ``away_league_code`` names the away side's league when it differs from
        the home side's; ``None`` means the same one. Which model that implies
        is the Space's decision, not this client's — it holds the corpus.
        """
        if not self.config.inference.enabled:
            raise RuntimeError("On-demand inference is disabled in configuration")

        async with httpx.AsyncClient(timeout=self.PREDICTION_TIMEOUT) as client:
            response = await client.post(
                f"{self._space_url()}/predict-custom",
                json={
                    "home_team": home_team,
                    "away_team": away_team,
                    "league_code": league_code,
                    "away_league_code": away_league_code,
                },
            )
        if response.status_code == self.REFUSED_STATUS:
            raise PredictionRefused(self._refusal_reason(response))
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def _refusal_reason(self, response: Any) -> str:
        """The reason the Space stated, or an honest admission there was none.

        ``raise_for_status`` would replace it with the status line, discarding
        the one part of the answer the user can act on.
        """
        try:
            detail = response.json().get("detail")
        except Exception:
            return self.UNSTATED_REFUSAL
        return str(detail) if detail else self.UNSTATED_REFUSAL

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
