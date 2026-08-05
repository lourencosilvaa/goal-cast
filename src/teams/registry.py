"""Reader for the shipped canonical team registry.

The registry — ``{league_code: [team, ...]}`` in the football-data.co.uk
spelling — is the authority on what a team may be *called* in this project.
The backend already serves it through ``StaticTeamRepository``; this is the
synchronous reader the offline pipeline and the name resolver use.

Like every other source in the project it degrades to an empty mapping rather
than raising, so a missing file never takes a caller down.
"""

import json
from pathlib import Path
from typing import Final

ENCODING: Final[str] = "utf-8"


def load_team_registry(path: str | Path) -> dict[str, list[str]]:
    """Return ``{league_code: [team, ...]}``, or ``{}`` when unavailable."""
    try:
        raw = json.loads(Path(path).read_text(encoding=ENCODING))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(code): [str(team) for team in teams]
        for code, teams in raw.items()
        if isinstance(teams, list)
    }
