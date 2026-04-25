"""Leagues endpoint — lists available leagues."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["leagues"])


@router.get("/leagues")
async def list_leagues(request: Request) -> list[dict[str, str]]:
    """Return all supported leagues."""
    config = request.app.state.config
    return [{"code": code, "name": name} for code, name in config.data.leagues.items()]
