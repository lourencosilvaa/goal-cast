"""Phase 1 tests for the HuggingFace Space restart script.

The Space loads its historical frame once at boot, so a freshly uploaded
Parquet snapshot only reaches users after a restart. These tests pin that
behaviour without touching the network: the Hub client is always injected,
and every repo id / token used here is set explicitly per test rather than
read from the ambient environment.
"""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config_loader import HuggingFaceConfig  # noqa: E402
from scripts.restart_hf_space import (  # noqa: E402
    HuggingFaceSpaceRestarter,
    SpaceRestartSpec,
    _build_client,
    _resolve_spec,
    _run,
    main,
)

_SPACE_REPO = "tester/goal-cast-space"
_TOKEN = "hf_test_token"
_CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "config" / "config.yaml")


class FakeHubClient:
    """Records restart calls in place of ``huggingface_hub.HfApi``."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    def restart_space(self, repo_id: str) -> None:
        self.calls.append(repo_id)
        if self.error is not None:
            raise self.error


def _args(**overrides) -> argparse.Namespace:
    values = {
        "config": _CONFIG_PATH,
        "space_repo_id": "",
        "token": "",
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestResolveSpec:

    def test_takes_values_from_config(self):
        hf = HuggingFaceConfig(space_repo_id=_SPACE_REPO, hf_token=_TOKEN)
        spec = _resolve_spec(_args(), hf)
        assert spec == SpaceRestartSpec(repo_id=_SPACE_REPO, token=_TOKEN)

    def test_cli_argument_overrides_config(self):
        hf = HuggingFaceConfig(space_repo_id="tester/from-config", hf_token=_TOKEN)
        spec = _resolve_spec(_args(space_repo_id=_SPACE_REPO), hf)
        assert spec.repo_id == _SPACE_REPO

    def test_cli_token_overrides_config(self):
        hf = HuggingFaceConfig(space_repo_id=_SPACE_REPO, hf_token="from-config")
        spec = _resolve_spec(_args(token=_TOKEN), hf)
        assert spec.token == _TOKEN


class TestRun:

    def test_restarts_the_configured_space_once(self):
        client = FakeHubClient()
        _run(
            SpaceRestartSpec(repo_id=_SPACE_REPO, token=_TOKEN),
            dry_run=False,
            client=client,
        )
        assert client.calls == [_SPACE_REPO]

    def test_missing_repo_id_exits_non_zero(self):
        client = FakeHubClient()
        with pytest.raises(SystemExit) as exc:
            _run(SpaceRestartSpec(repo_id="", token=_TOKEN), dry_run=False, client=client)
        assert exc.value.code != 0
        assert client.calls == []

    def test_missing_token_exits_non_zero(self):
        client = FakeHubClient()
        with pytest.raises(SystemExit) as exc:
            _run(
                SpaceRestartSpec(repo_id=_SPACE_REPO, token=""),
                dry_run=False,
                client=client,
            )
        assert exc.value.code != 0
        assert client.calls == []

    def test_dry_run_does_not_call_the_hub(self, capsys):
        client = FakeHubClient()
        _run(
            SpaceRestartSpec(repo_id=_SPACE_REPO, token=_TOKEN),
            dry_run=True,
            client=client,
        )
        assert client.calls == []
        assert _SPACE_REPO in capsys.readouterr().out

    def test_client_failure_exits_non_zero(self):
        client = FakeHubClient(error=RuntimeError("space is building"))
        with pytest.raises(SystemExit) as exc:
            _run(
                SpaceRestartSpec(repo_id=_SPACE_REPO, token=_TOKEN),
                dry_run=False,
                client=client,
            )
        assert exc.value.code != 0


class TestHuggingFaceSpaceRestarter:

    def test_delegates_to_the_injected_client(self):
        client = FakeHubClient()
        restarter = HuggingFaceSpaceRestarter(
            SpaceRestartSpec(repo_id=_SPACE_REPO, token=_TOKEN), client
        )
        restarter.restart()
        assert client.calls == [_SPACE_REPO]


class TestRunEdgeCases:
    """Phase 3: boundaries and error paths found while implementing."""

    def test_dry_run_needs_no_token(self, capsys):
        """A preview must work on a machine that holds no credentials."""
        _run(
            SpaceRestartSpec(repo_id=_SPACE_REPO, token=""),
            dry_run=True,
            client=FakeHubClient(),
        )
        assert "dry-run" in capsys.readouterr().out

    def test_dry_run_still_requires_a_repo_id(self):
        with pytest.raises(SystemExit) as exc:
            _run(SpaceRestartSpec(repo_id="", token=_TOKEN), dry_run=True, client=None)
        assert exc.value.code != 0

    def test_success_message_names_the_space(self, capsys):
        _run(
            SpaceRestartSpec(repo_id=_SPACE_REPO, token=_TOKEN),
            dry_run=False,
            client=FakeHubClient(),
        )
        assert _SPACE_REPO in capsys.readouterr().out

    def test_failure_message_names_the_space_and_cause(self, capsys):
        client = FakeHubClient(error=RuntimeError("space is building"))
        with pytest.raises(SystemExit):
            _run(
                SpaceRestartSpec(repo_id=_SPACE_REPO, token=_TOKEN),
                dry_run=False,
                client=client,
            )
        output = capsys.readouterr().out
        assert _SPACE_REPO in output
        assert "space is building" in output

    def test_builds_its_own_client_when_none_is_injected(self, monkeypatch):
        """Production path: no client passed, so one is built from the spec."""
        client = FakeHubClient()
        built_with: list[SpaceRestartSpec] = []

        def fake_build(spec: SpaceRestartSpec):
            built_with.append(spec)
            return client

        monkeypatch.setattr("scripts.restart_hf_space._build_client", fake_build)
        spec = SpaceRestartSpec(repo_id=_SPACE_REPO, token=_TOKEN)
        _run(spec, dry_run=False)
        assert built_with == [spec]
        assert client.calls == [_SPACE_REPO]


class TestBuildClient:

    def test_builds_a_hub_client_carrying_the_token(self):
        """Constructing the client is offline; only its calls hit the network."""
        client = _build_client(SpaceRestartSpec(repo_id=_SPACE_REPO, token=_TOKEN))
        assert client.token == _TOKEN
        assert hasattr(client, "restart_space")


class TestSpaceRestartSpec:

    def test_is_immutable(self):
        spec = SpaceRestartSpec(repo_id=_SPACE_REPO, token=_TOKEN)
        with pytest.raises(Exception):
            spec.repo_id = "tester/other-space"  # type: ignore[misc]

    def test_resolves_to_empty_when_nothing_is_configured(self):
        spec = _resolve_spec(_args(), HuggingFaceConfig(space_repo_id="", hf_token=""))
        assert spec == SpaceRestartSpec(repo_id="", token="")


class TestMain:

    def test_wires_arguments_through_to_the_client(self, monkeypatch):
        client = FakeHubClient()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "restart_hf_space.py",
                "--config",
                _CONFIG_PATH,
                "--space-repo-id",
                _SPACE_REPO,
                "--token",
                _TOKEN,
            ],
        )
        main(client=client)
        assert client.calls == [_SPACE_REPO]

    def test_dry_run_leaves_the_hub_untouched(self, monkeypatch, capsys):
        client = FakeHubClient()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "restart_hf_space.py",
                "--config",
                _CONFIG_PATH,
                "--space-repo-id",
                _SPACE_REPO,
                "--dry-run",
            ],
        )
        main(client=client)
        assert client.calls == []
        assert "dry-run" in capsys.readouterr().out
