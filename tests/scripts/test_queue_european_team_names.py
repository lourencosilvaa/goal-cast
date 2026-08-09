"""Tests for the European name-queueing CLI.

The rule this file exists to enforce: the queue must never report success it
cannot see. Both ``record_pending`` and the sink swallow failures by design —
right for the fixture pipeline, which meets the same unknown name every run
and must not die when the queue is down; wrong for a deliberate admin action,
where "queued 101 names" while writing nothing is the worst of both.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SCRIPT = (
    Path(__file__).parent.parent.parent / "scripts" / "queue_european_team_names.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("queue_european_team_names", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["queue_european_team_names"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


class TestQueuedNames:
    def test_reads_the_queue(self, script):
        service = MagicMock()
        service.list_pending.return_value = [
            {"raw_name": "Sport Lisboa e Benfica"},
            {"raw_name": "Arsenal FC"},
        ]
        assert script.queued_names(service) == {"Sport Lisboa e Benfica", "Arsenal FC"}

    def test_unreadable_queue_yields_an_empty_set(self, script):
        """An unreadable queue must look empty, not crash the run."""
        service = MagicMock()
        service.list_pending.side_effect = RuntimeError("no such table")
        assert script.queued_names(service) == set()

    def test_empty_queue(self, script):
        service = MagicMock()
        service.list_pending.return_value = []
        assert script.queued_names(service) == set()


class TestBuildSink:
    def test_missing_credentials_are_named(self, script, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "key")
        sink, service, reason = script.build_sink()
        assert sink is None and service is None
        assert "SUPABASE_URL" in reason

    def test_both_missing_credentials_are_named(self, script, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        _, _, reason = script.build_sink()
        assert "SUPABASE_URL" in reason and "SUPABASE_SERVICE_KEY" in reason

    def test_reason_mentions_where_it_looked(self, script, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        _, _, reason = script.build_sink()
        assert ".env" in reason


class TestDiagnose:
    def test_unreadable_queue_is_reported_as_such(self, script):
        service = MagicMock()
        service.list_pending.side_effect = RuntimeError("permission denied")
        assert "unreadable" in script._diagnose(service)

    def test_readable_queue_points_at_the_missing_table(self, script):
        """The real cause when the table was never created."""
        service = MagicMock()
        service.list_pending.return_value = []
        assert "team_aliases" in script._diagnose(service)


class TestCorpusSelection:
    def _config(self, script):
        from config.config_loader import Config, load_config

        config: Config = load_config("config/config.yaml")
        return config

    def test_main_draws_by_default(self, script):
        config = self._config(script)
        assert script.corpus_path(config, False) == config.european.cache_path

    def test_qualifiers_when_requested(self, script):
        config = self._config(script)
        assert (
            script.corpus_path(config, True)
            == config.european.qualifiers_cache_path
        )
