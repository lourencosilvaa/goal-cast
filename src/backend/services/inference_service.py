"""On-demand prediction service: delegates inference to the HF Space API."""

from typing import Any

import requests


class InferenceService:
    """Thin HTTP client that calls the HuggingFace Space inference endpoint."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def run(
        self,
        target_date: str | None = None,
        league_codes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.config.inference.enabled:
            raise RuntimeError("On-demand inference is disabled in configuration")

        space_url = self.config.inference.space_url.rstrip("/")
        if not space_url:
            raise RuntimeError(
                "HF Space URL is not configured. Set the HF_SPACE_URL environment variable."
            )
        response = requests.post(
            f"{space_url}/infer",
            json={"date": target_date, "league_codes": league_codes},
            timeout=120,
        )
        response.raise_for_status()
        data: list[dict[str, Any]] = response.json().get("predictions", [])
        return data

    def predict_custom(
        self,
        home_team: str,
        away_team: str,
        league_code: str,
    ) -> dict[str, Any]:
        if not self.config.inference.enabled:
            raise RuntimeError("On-demand inference is disabled in configuration")

        space_url = self.config.inference.space_url.rstrip("/")
        if not space_url:
            raise RuntimeError(
                "HF Space URL is not configured. Set the HF_SPACE_URL environment variable."
            )
        response = requests.post(
            f"{space_url}/predict-custom",
            json={
                "home_team": home_team,
                "away_team": away_team,
                "league_code": league_code,
            },
            timeout=120,
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result
