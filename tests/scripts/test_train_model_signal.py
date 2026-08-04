"""Tests for the CI retrained-signal helper in train_model.

The helper writes a GitHub Actions step output so the workflow can skip the
upload + redeploy steps when no retraining happened. Outside GitHub Actions
(no GITHUB_OUTPUT) it must be a silent no-op.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train_model import _emit_retrained


class TestEmitRetrained:
    def test_writes_true(self, tmp_path, monkeypatch) -> None:
        out = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        _emit_retrained(True)
        assert out.read_text() == "retrained=true\n"

    def test_writes_false(self, tmp_path, monkeypatch) -> None:
        out = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        _emit_retrained(False)
        assert out.read_text() == "retrained=false\n"

    def test_appends_not_overwrites(self, tmp_path, monkeypatch) -> None:
        out = tmp_path / "gh_output"
        out.write_text("existing=1\n")
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        _emit_retrained(True)
        assert out.read_text() == "existing=1\nretrained=true\n"

    def test_noop_without_env(self, monkeypatch) -> None:
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        # Must not raise when not running under GitHub Actions.
        _emit_retrained(True)
