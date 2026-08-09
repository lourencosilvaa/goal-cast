"""Local checkout of the openfootball data repository.

This is the only piece of the European track that touches the network, and it
is driven by an explicit CLI step (``scripts/build_european_corpus.py``) rather
than by training. That separation is deliberate: if a model refit could reach
for GitHub, an outage would produce a model with no cross-league links at all
while looking exactly like a healthy run.

The data is public domain and small — the whole 15-season history is well
under a megabyte — so a shallow clone is cheap to take and cheap to refresh.
"""

import subprocess
from pathlib import Path
from typing import ClassVar

from config.config_loader import EuropeanConfig


class OpenFootballRepository:
    """Clones or updates the openfootball checkout.

    Degrades rather than raises: a failed sync leaves whatever is already on
    disk in place, so a corpus build can still run from the last good copy.
    """

    #: Only the current tip is needed — no consumer reads history.
    CLONE_DEPTH: ClassVar[str] = "1"

    #: Seconds before a git call is abandoned. Guards against a hung network
    #: call silently stalling a scheduled build.
    TIMEOUT_SECONDS: ClassVar[int] = 120

    def __init__(self, config: EuropeanConfig) -> None:
        self.config = config
        self._path = Path(config.checkout_path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_checked_out(self) -> bool:
        return (self._path / ".git").is_dir()

    def sync(self) -> bool:
        """Clone or pull the repository. ``True`` when git reported success."""
        command = self._pull_command() if self.is_checked_out else self._clone_command()
        if not self.is_checked_out:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def _clone_command(self) -> list[str]:
        return [
            "git",
            "clone",
            "--depth",
            self.CLONE_DEPTH,
            self.config.repo_url,
            str(self._path),
        ]

    def _pull_command(self) -> list[str]:
        return ["git", "-C", str(self._path), "pull", "--ff-only"]
