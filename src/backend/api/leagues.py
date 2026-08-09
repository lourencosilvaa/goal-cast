"""Leagues endpoint — lists available leagues and competitions."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from src.backend.core.auth import get_approved_user

router = APIRouter(prefix="/api", tags=["leagues"])

#: Domestic divisions, with a football-data.co.uk feed and a league table.
TYPE_LEAGUE = "league"
#: UEFA club competitions. No feed, no table — their history comes from the
#: openfootball corpus and their fixtures from the provider chain.
TYPE_COMPETITION = "competition"


@router.get("/leagues")
async def list_leagues(
    request: Request,
    _: Annotated[str, Depends(get_approved_user)],
) -> list[dict[str, str]]:
    """Every selectable league and competition.

    Composed from two sources rather than one. The European competitions are
    deliberately *not* in ``data.leagues``: that mapping drives the data
    loader, which fetches a football-data.co.uk CSV per code, and no such feed
    exists for the Champions League — the absence that the openfootball corpus
    exists to fill. Listing them there would make the loader request a missing
    file on every run.

    ``type`` tells the UI which pipeline a code belongs to, so it can avoid
    offering a league table for a competition that has none.
    """
    config = request.app.state.config
    # `leagues` is the training corpus, which is wider than the product on
    # purpose — see DataConfig. Only `served_leagues` may be offered.
    served = set(config.data.served_leagues)
    entries = [
        {"code": code, "name": name, "type": TYPE_LEAGUE}
        for code, name in config.data.leagues.items()
        if code in served
    ]
    # Not filtered: the UEFA competitions are absent from `data.leagues`
    # entirely, so a subset of it can say nothing about them.
    entries.extend(_european_entries(config))
    return entries


def _european_entries(config: Any) -> list[dict[str, str]]:
    """The UEFA competitions, or nothing when the track is disabled."""
    european = getattr(config, "european", None)
    if european is None or not european.enabled:
        return []
    names = getattr(european, "competition_names", {}) or {}
    return [
        {
            "code": code,
            # Fall back to the code itself: a competition without a display
            # name is still selectable, just tersely labelled.
            "name": names.get(code, code),
            "type": TYPE_COMPETITION,
        }
        for code in european.competitions
    ]
