"""On-demand prediction service: delegates inference to the HF Space API."""

import time
from typing import Any

import requests


class InferenceService:
    """Thin HTTP client that calls the HuggingFace Space inference endpoint."""

    _WAKE_RETRIES = 10
    _WAKE_INTERVAL = 15  # seconds between health checks while Space is waking

    def __init__(self, config: Any) -> None:
        self.config = config

    def _space_url(self) -> str:
        url = self.config.inference.space_url.rstrip("/")
        if not url:
            raise RuntimeError(
                "HF Space URL is not configured. Set the HF_SPACE_URL environment variable."
            )
        return url

    def _headers(self) -> dict[str, str]:
        token = getattr(self.config.huggingface, "hf_token", "")
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    def _wake_space(self) -> None:
        """Poll /health until the Space is up or retries are exhausted."""
        url = f"{self._space_url()}/health"
        for attempt in range(self._WAKE_RETRIES):
            try:
                r = requests.get(
                    url, headers=self._headers(), timeout=30, allow_redirects=True
                )
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            if attempt < self._WAKE_RETRIES - 1:
                time.sleep(self._WAKE_INTERVAL)
        raise RuntimeError("HF Space did not become ready after wake-up retries")

    def run(
        self,
        target_date: str | None = None,
        league_codes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.config.inference.enabled:
            raise RuntimeError("On-demand inference is disabled in configuration")

        self._wake_space()
        response = requests.post(
            f"{self._space_url()}/infer",
            json={"date": target_date, "league_codes": league_codes},
            headers=self._headers(),
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

        self._wake_space()
        response = requests.post(
            f"{self._space_url()}/predict-custom",
            json={
                "home_team": home_team,
                "away_team": away_team,
                "league_code": league_code,
            },
            headers=self._headers(),
            timeout=120,
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result
