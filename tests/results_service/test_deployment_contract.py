"""Static checks on the results service's image and its workflow.

Two failure modes this catches, both silent in CI and both expensive:

* **A package the service imports is not COPYed into the image.** The build
  succeeds, the container starts, and the first import fails at run time.
* **A directory in the image is missing from the workflow's path filter.** A
  change to it is pushed, no build is triggered, and Render keeps serving the
  previous code with a green tick beside the commit.

The same guard as ``tests/backend/test_dockerfile_contract.py``, applied to the
second image.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = PROJECT_ROOT / "Dockerfile.results"
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "deploy-results.yml"
VENV_BIN = "/app/.venv/bin"


def _lines() -> list[str]:
    return DOCKERFILE.read_text().splitlines()


def _copied_paths() -> list[str]:
    return [
        line.split()[1]
        for line in _lines()
        if line.startswith("COPY ") and not line.startswith("COPY --from")
    ]


def _cmd_line() -> str:
    matches = [line for line in _lines() if line.startswith("CMD")]
    assert len(matches) == 1, f"Expected exactly one CMD, found {len(matches)}"
    return matches[0]


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _trigger_paths() -> list[str]:
    # PyYAML reads the bare key `on:` as the boolean True.
    triggers = _workflow()[True]
    return list(triggers["push"]["paths"])


class TestImageContents:
    def test_the_service_package_is_copied(self):
        assert "src/results_service/" in _copied_paths()

    def test_the_provider_library_is_copied(self):
        """``src.scrapers.results`` is what the service composes."""
        assert "src/scrapers/" in _copied_paths()

    def test_the_shared_contract_is_copied(self):
        """The response models the gateway parses on the other side."""
        assert "src/contracts/" in _copied_paths()

    def test_the_config_package_is_copied(self):
        assert "config/" in _copied_paths()

    def test_the_src_package_marker_is_copied(self):
        """Without it, ``src.`` is not a package and every import fails."""
        assert "src/__init__.py" in _copied_paths()

    def test_the_frontend_is_not_copied(self):
        assert not any(path.startswith("src/frontend") for path in _copied_paths())

    def test_the_backend_is_not_copied(self):
        """The two services share a contract, not a codebase."""
        assert not any(path.startswith("src/backend") for path in _copied_paths())


class TestRuntimeContract:
    def test_only_the_results_dependency_group_is_installed(self):
        sync_lines = [line for line in _lines() if "uv sync" in line]
        assert len(sync_lines) == 1
        assert "--only-group results" in sync_lines[0]
        assert "--no-dev" in sync_lines[0]

    def test_the_command_does_not_invoke_uv_at_runtime(self):
        """`uv run` would re-sync against the default groups on every cold
        start and pull the whole ML toolchain into a results container."""
        assert "uv" not in _cmd_line().split('"')

    def test_the_command_starts_the_service_entry_point(self):
        assert "src.results_service.main:app" in _cmd_line()

    def test_the_venv_is_on_the_path(self):
        assert any(
            line.startswith("ENV PATH=") and VENV_BIN in line for line in _lines()
        )

    def test_chromium_is_installed_for_the_fallback(self):
        """A kill switch that needs a rebuild to flip is not a kill switch."""
        assert any("playwright install" in line for line in _lines())


class TestDependencyGroup:
    def _group(self) -> list[str]:
        import tomllib

        pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
        return pyproject["dependency-groups"]["results"]

    def test_the_group_exists(self):
        assert self._group()

    def test_the_group_carries_playwright(self):
        assert any(dep.startswith("playwright") for dep in self._group())

    def test_the_group_carries_no_machine_learning_stack(self):
        heavy = ("pandas", "numpy", "scikit-learn", "xgboost", "scipy", "pyarrow")
        assert not [dep for dep in self._group() if dep.startswith(heavy)]

    def test_the_backend_group_does_not_gain_playwright(self):
        """The split exists to keep Chromium out of the application image."""
        import tomllib

        pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
        backend = pyproject["dependency-groups"]["backend"]
        assert not [dep for dep in backend if dep.startswith("playwright")]


class TestWorkflowTriggers:
    def test_every_copied_source_directory_triggers_a_build(self):
        """The test that prevents a stale image: a directory that ends up in
        the container but not in the filter is code that ships only by
        accident, on the next unrelated push."""
        paths = _trigger_paths()
        for copied in _copied_paths():
            if not copied.startswith(("src/", "config/")):
                continue
            root = "/".join(copied.rstrip("/").split("/")[:2])
            assert any(
                pattern.rstrip("/*").rstrip("/") in (root, copied.rstrip("/"))
                for pattern in paths
            ), f"{copied} is copied into the image but triggers no build"

    def test_the_dockerfile_itself_triggers_a_build(self):
        assert "Dockerfile.results" in _trigger_paths()

    def test_the_lockfile_triggers_a_build(self):
        assert "uv.lock" in _trigger_paths()

    def test_the_workflow_can_be_run_by_hand(self):
        """Needed for the very first build, before Render has an image."""
        assert "workflow_dispatch" in _workflow()[True]

    def test_the_image_name_does_not_collide_with_the_application_image(self):
        assert "-results" in WORKFLOW.read_text()

    def test_the_build_cache_scope_is_its_own(self):
        """Sharing the app's scope would have a Chromium layer evicting the
        frontend's node_modules on every other build."""
        body = WORKFLOW.read_text()
        assert "scope=results" in body
        assert "scope=app" not in body

    def test_the_deploy_hook_is_a_separate_secret(self):
        assert "RENDER_RESULTS_DEPLOY_HOOK_URL" in WORKFLOW.read_text()


class TestApplicationWorkflowUnchanged:
    def test_the_application_workflow_does_not_build_this_image(self):
        body = (PROJECT_ROOT / ".github" / "workflows" / "deploy.yml").read_text()
        assert "Dockerfile.results" not in body

    def test_the_application_image_does_not_carry_the_provider_library(self):
        """Importing a provider into the backend would put Playwright back in
        the image the split removed it from."""
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text().splitlines()
        copied = [
            line.split()[1]
            for line in dockerfile
            if line.startswith("COPY ") and not line.startswith("COPY --from")
        ]
        assert "src/scrapers/" not in copied
