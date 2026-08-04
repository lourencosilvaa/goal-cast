"""Guards the Dockerfile runtime contract.

``uv run`` synchronises the project environment before executing.  The build
stage installs only the ``backend`` group, so a runtime ``uv run`` re-resolves
against the default groups and re-downloads the full ML + dev toolchain on
every container start — the cause of multi-minute Render cold starts.
The container must therefore invoke the already-installed venv directly.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_BIN = "/app/.venv/bin"


def _dockerfile_lines() -> list[str]:
    return (PROJECT_ROOT / "Dockerfile").read_text().splitlines()


def _cmd_line() -> str:
    matches = [line for line in _dockerfile_lines() if line.startswith("CMD")]
    assert len(matches) == 1, f"Expected exactly one CMD, found {len(matches)}"
    return matches[0]


class TestDockerfileRuntimeContract:

    def test_cmd_does_not_invoke_uv_at_runtime(self):
        assert "uv" not in _cmd_line().split('"')

    def test_cmd_starts_uvicorn(self):
        assert "uvicorn" in _cmd_line()

    def test_venv_bin_is_on_path(self):
        env_lines = [
            line for line in _dockerfile_lines() if line.startswith("ENV PATH=")
        ]
        assert any(VENV_BIN in line for line in env_lines), (
            f"Dockerfile must put {VENV_BIN} on PATH so the CMD resolves "
            "uvicorn from the environment built by `uv sync`."
        )

    def test_build_stage_still_installs_backend_group_only(self):
        sync_lines = [line for line in _dockerfile_lines() if "uv sync" in line]
        assert len(sync_lines) == 1
        assert "--only-group backend" in sync_lines[0]
        assert "--no-dev" in sync_lines[0]
