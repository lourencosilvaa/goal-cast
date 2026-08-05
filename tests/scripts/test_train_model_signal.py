"""Tests for the CI signal helpers in train_model.

Two independent signals drive the workflow: `retrained` says the model was
refit (artefact upload + Render redeploy), `data_changed` says new match data
arrived (dataset upload + Space restart). They must be able to disagree — a
held-back refit still ships fresh data. Outside GitHub Actions (no
GITHUB_OUTPUT) both must be silent no-ops.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train_model import _emit_data_changed, _emit_retrained


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


class TestEmitDataChanged:
    def test_writes_true(self, tmp_path, monkeypatch) -> None:
        out = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        _emit_data_changed(True)
        assert out.read_text() == "data_changed=true\n"

    def test_writes_false(self, tmp_path, monkeypatch) -> None:
        out = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        _emit_data_changed(False)
        assert out.read_text() == "data_changed=false\n"

    def test_noop_without_env(self, monkeypatch) -> None:
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        _emit_data_changed(True)


class TestSignalsAreIndependent:
    def test_both_signals_share_one_output_file(self, tmp_path, monkeypatch) -> None:
        """The held-back-refit case: fresh data, no new model."""
        out = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        _emit_data_changed(True)
        _emit_retrained(False)
        assert out.read_text() == "data_changed=true\nretrained=false\n"


class TestCountNewMatches:
    """Phase 3: the count that feeds the threshold."""

    @staticmethod
    def _frame():
        import pandas as pd

        return pd.DataFrame(
            {
                "Date": ["09/05/2026", "16/05/2026", "22/08/2026"],
                "HomeTeam": ["A", "B", "C"],
                "AwayTeam": ["B", "C", "A"],
            }
        )

    def test_counts_only_matches_after_the_training_date(self):
        import pandas as pd

        from scripts.train_model import _count_new_matches

        assert _count_new_matches(self._frame(), pd.Timestamp("2026-05-16")) == 1

    def test_untrained_model_has_seen_nothing(self):
        from scripts.train_model import _count_new_matches

        assert _count_new_matches(self._frame(), None) == 3

    def test_returns_zero_without_a_date_column(self):
        import pandas as pd

        from scripts.train_model import _count_new_matches

        frame = pd.DataFrame({"HomeTeam": ["A"], "AwayTeam": ["B"]})
        assert _count_new_matches(frame, pd.Timestamp("2026-05-16")) == 0

    def test_unparseable_dates_are_not_counted(self):
        import pandas as pd

        from scripts.train_model import _count_new_matches

        frame = pd.DataFrame({"Date": ["not-a-date"], "HomeTeam": ["A"], "AwayTeam": ["B"]})
        assert _count_new_matches(frame, pd.Timestamp("2026-05-16")) == 0
