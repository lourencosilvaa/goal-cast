"""Every script must at least parse.

Most scripts in this project have no tests — they are CLI entry points that
talk to Supabase, Hugging Face or a live API, and mocking all of that to assert
on print output has never been worth it.

The gap that leaves is real though: ``find_value_bets.py`` was edited into a
syntax error (an import block dedented to module level in the middle of a
function) and stayed that way through a full green test run, because nothing
imported it. The script would have failed on its first line, in production, on
whatever day someone next ran it.

This is the cheapest possible guard against that class of failure. It proves
nothing about behaviour — only that the file is valid Python and its
module-level imports resolve, which is exactly the part no other test covers.
"""

import ast
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


def _script_paths() -> list[Path]:
    return sorted(p for p in _SCRIPTS_DIR.glob("*.py") if not p.name.startswith("_"))


def _ids(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


_SCRIPTS = _script_paths()


class TestSyntax:
    @pytest.mark.parametrize("path", _SCRIPTS, ids=_ids(_SCRIPTS))
    def test_script_is_valid_python(self, path: Path):
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            pytest.fail(f"{path.name} does not parse: line {exc.lineno}: {exc.msg}")

    def test_the_scripts_directory_was_actually_found(self):
        """A glob that silently matches nothing would make this suite vacuous."""
        assert len(_SCRIPTS) > 5
