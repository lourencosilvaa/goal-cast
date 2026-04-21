"""Leagues endpoint — lists available leagues."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["leagues"])


@router.get("/leagues")
async def list_leagues(request: Request) -> list[dict[str, str]]:
    """Return all supported leagues (domestic + org competitions when API key set)."""
    config = request.app.state.config
    leagues = [
        {"code": code, "name": name} for code, name in config.data.leagues.items()
    ]
    if config.football_data_org.api_key:
        for code, name in config.football_data_org.competitions.items():
            leagues.append({"code": code, "name": name})
    return leagues
