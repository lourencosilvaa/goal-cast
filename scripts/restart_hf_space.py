"""
Restart the HuggingFace Space so it reloads the freshly uploaded dataset.

The Space builds its match history once, in ``lifespan``, from the per-league
Parquet snapshot on the Hub. Uploading a newer snapshot therefore changes
nothing for a Space that is already running: it keeps answering from the frame
it booted with, and recent results silently never reach the UI. The retrain
workflow calls this script right after the upload, so the Space comes back
holding the data that was just published.

Usage:
    uv run python scripts/restart_hf_space.py
    uv run python scripts/restart_hf_space.py --space-repo-id myuser/my-space
    uv run python scripts/restart_hf_space.py --dry-run

Environment variables (consumed through the config loader):
    HF_TOKEN          — HuggingFace token with write access to the Space
    HF_SPACE_REPO_ID  — Space repo ID, e.g. myuser/football-prediction
"""

import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import HuggingFaceConfig, load_config  # noqa: E402


@dataclass(frozen=True)
class SpaceRestartSpec:
    """Everything a restarter needs, so no argument list has to be threaded."""

    repo_id: str
    token: str


class SpaceRestarter(ABC):
    """Triggers a restart of the deployed inference service."""

    @abstractmethod
    def restart(self) -> None:
        """Restart the service, raising on failure."""


class HuggingFaceSpaceRestarter(SpaceRestarter):
    """Restarts a Space through an injected ``huggingface_hub.HfApi`` client."""

    def __init__(self, spec: SpaceRestartSpec, client: Any) -> None:
        self._spec = spec
        self._client = client

    def restart(self) -> None:
        self._client.restart_space(repo_id=self._spec.repo_id)


class DryRunSpaceRestarter(SpaceRestarter):
    """Reports the target without touching the Hub."""

    def __init__(self, spec: SpaceRestartSpec) -> None:
        self._spec = spec

    def restart(self) -> None:
        print(f"[dry-run] Would restart Space: {self._spec.repo_id}")


#: Exit status used for every failure, so the workflow step goes red instead of
#: leaving the Space quietly serving stale data.
_FAILURE_EXIT_CODE = 1


def _resolve_spec(
    args: argparse.Namespace, hf_config: HuggingFaceConfig
) -> SpaceRestartSpec:
    """Command-line arguments win over the configured (env-backed) values."""
    return SpaceRestartSpec(
        repo_id=args.space_repo_id or hf_config.space_repo_id,
        token=args.token or hf_config.hf_token,
    )


def _build_client(spec: SpaceRestartSpec) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=spec.token)


def _run(spec: SpaceRestartSpec, dry_run: bool, client: Any | None = None) -> None:
    """Restart the Space described by ``spec``, exiting non-zero on failure."""
    if not spec.repo_id:
        print("ERROR: --space-repo-id or HF_SPACE_REPO_ID env var is required")
        sys.exit(_FAILURE_EXIT_CODE)

    if dry_run:
        DryRunSpaceRestarter(spec).restart()
        return

    if not spec.token:
        print("ERROR: --token or HF_TOKEN env var is required")
        sys.exit(_FAILURE_EXIT_CODE)

    hub = client if client is not None else _build_client(spec)
    restarter = HuggingFaceSpaceRestarter(spec, hub)
    try:
        restarter.restart()
    except Exception as exc:
        print(f"ERROR: could not restart Space {spec.repo_id}: {exc}")
        sys.exit(_FAILURE_EXIT_CODE)

    print(f"✓ Restart requested → https://huggingface.co/spaces/{spec.repo_id}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restart the HuggingFace Space")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config file providing the Space repo ID and token",
    )
    parser.add_argument(
        "--space-repo-id",
        default="",
        help="Space repo ID, e.g. myuser/football-prediction",
    )
    parser.add_argument(
        "--token",
        default="",
        help="HuggingFace API token with write access to the Space",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the target Space without restarting it",
    )
    return parser.parse_args()


def main(client: Any | None = None) -> None:
    args = _parse_args()
    config = load_config(args.config)
    _run(_resolve_spec(args, config.huggingface), dry_run=args.dry_run, client=client)


if __name__ == "__main__":
    main()
