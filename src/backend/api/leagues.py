"""Leagues endpoint — lists available leagues."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from src.backend.core.auth import get_approved_user

router = APIRouter(prefix="/api", tags=["leagues"])


@router.get("/leagues")
async def list_leagues(
    request: Request,
    _: Annotated[str, Depends(get_approved_user)],
) -> list[dict[str, str]]:
    """Return all supported leagues."""
    config = request.app.state.config
    return [{"code": code, "name": name} for code, name in config.data.leagues.items()]
