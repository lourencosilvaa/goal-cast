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


class TestDockerfileSourceContract:
    """Every package the backend imports must be copied into the image.

    The build installs only the ``backend`` dependency group and copies a
    hand-picked set of directories, so a new top-level package is a silent
    ImportError at container start rather than a build failure.
    """

    @staticmethod
    def _copied_paths() -> list[str]:
        return [
            line.split()[1]
            for line in _dockerfile_lines()
            if line.startswith("COPY ") and not line.startswith("COPY --from")
        ]

    def test_backend_package_is_copied(self):
        assert "src/backend/" in self._copied_paths()

    def test_config_package_is_copied(self):
        assert "config/" in self._copied_paths()

    def test_shared_teams_package_is_copied(self):
        """``src.teams`` backs the admin alias screen (src/backend/api/team_aliases.py)."""
        assert "src/teams/" in self._copied_paths()

    def test_the_results_contract_is_copied(self):
        """``src.contracts.results`` is imported by the results gateway, which
        ``src/backend/main.py`` imports at start-up. Missing, the container
        does not merely lose /api/results — it never boots."""
        assert "src/contracts/" in self._copied_paths()

    def test_every_src_package_the_backend_imports_is_copied(self):
        """Catches the next one of these rather than this one.

        A new top-level package under ``src/`` reaches the backend as an
        ordinary import and passes every test on a machine with the repo
        checked out. The image has only what is listed here."""
        copied = self._copied_paths()
        backend = PROJECT_ROOT / "src" / "backend"
        imported = {
            line.split(".")[1].split()[0].rstrip(":")
            for path in backend.rglob("*.py")
            for line in path.read_text().splitlines()
            if line.startswith(("from src.", "import src."))
        }
        for package in sorted(imported):
            assert any(
                copied_path.startswith(f"src/{package}") for copied_path in copied
            ), f"src/{package} is imported by the backend but never COPYed"

    def test_the_outcome_model_is_copied(self):
        """``src.backend.services.in_play`` imports ``OutcomeProbabilities``.

        Without this the container starts and the in-play endpoint raises
        ImportError on its first call — a build that succeeds and a feature
        that does not."""
        assert "src/models/outcome_model.py" in self._copied_paths()

    def test_the_models_package_marker_is_copied(self):
        """Without it ``src.models`` is not a package and the import above
        fails even though the file is present."""
        assert "src/models/__init__.py" in self._copied_paths()

    def test_the_models_package_is_not_copied_wholesale(self):
        """Everything else under src/models/ imports pandas, sklearn or
        xgboost — the stack this image is built without."""
        assert "src/models/" not in self._copied_paths()


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


class TestWorkflowTriggers:
    """A directory inside the image but outside the filter ships by accident.

    The push builds nothing, Render keeps serving the previous code, and the
    commit carries a green tick — the same failure the results service is
    guarded against in ``tests/results_service/test_deployment_contract.py``.
    """

    @staticmethod
    def _trigger_paths() -> list[str]:
        import yaml

        workflow = yaml.safe_load(
            (PROJECT_ROOT / ".github" / "workflows" / "deploy.yml").read_text()
        )
        # PyYAML reads the bare key `on:` as the boolean True.
        return list(workflow[True]["push"]["paths"])

    def test_every_copied_source_directory_triggers_a_build(self):
        paths = self._trigger_paths()
        for copied in TestDockerfileSourceContract._copied_paths():
            if not copied.startswith(("src/", "config/")):
                continue
            root = "/".join(copied.rstrip("/").split("/")[:2])
            assert any(
                pattern.rstrip("/*").rstrip("/") in (root, copied.rstrip("/"))
                for pattern in paths
            ), f"{copied} is copied into the image but triggers no build"

    def test_the_dockerfile_itself_triggers_a_build(self):
        assert "Dockerfile" in self._trigger_paths()

    def test_the_lockfile_triggers_a_build(self):
        """`uv sync --frozen` installs from it, so a dependency change that
        touches nothing else must still rebuild."""
        assert "uv.lock" in self._trigger_paths()


class TestTeamsRegistryShipsInImage:
    """The static teams registry is the offline fallback for /teams — if the
    image does not carry it, the custom-prediction pickers go blank whenever
    the HF Space is unreachable."""

    def test_registry_file_exists_in_repo(self):
        from config.config_loader import load_config

        registry = PROJECT_ROOT / load_config(
            PROJECT_ROOT / "config" / "config.yaml"
        ).teams.registry_path
        assert registry.exists()

    def test_registry_is_under_a_copied_path(self):
        """It must live beneath a directory the Dockerfile actually COPYs."""
        from config.config_loader import load_config

        registry_path = load_config(
            PROJECT_ROOT / "config" / "config.yaml"
        ).teams.registry_path
        copied = [
            line.split()[1]
            for line in _dockerfile_lines()
            if line.startswith("COPY") and not line.startswith("COPY --from")
        ]
        assert any(registry_path.startswith(prefix) for prefix in copied), (
            f"{registry_path} is not covered by any COPY in the Dockerfile: {copied}"
        )
