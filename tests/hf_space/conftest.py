"""Imports ``hf_space/app.py`` the way the Space itself does.

The Space runs with ``hf_space/`` as its working directory, so its ``config``
and ``src`` packages shadow the main project's packages of the same name. In a
pytest process both are already imported, and a plain ``sys.path`` insert would
therefore bind ``from config.config_loader import SpaceConfig`` to the *main*
loader, which has no such class.

So the fixture unbinds both package trees for the duration of the import and
puts the process back exactly as it found it. Anything less leaks the Space's
modules into every test that runs afterwards.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Iterator

import pytest

SPACE_ROOT = Path(__file__).resolve().parents[2] / "hf_space"

#: Package roots the Space redefines. Both must go, or the halves mix.
_SHADOWED = ("src", "config")


def _purge_shadowed() -> None:
    for name in list(sys.modules):
        if name in _SHADOWED or name.startswith(tuple(f"{p}." for p in _SHADOWED)):
            del sys.modules[name]


#: The name the app module is imported under, so teardown can drop it.
_APP_MODULE = "space_app_under_test"


def _shadowed_modules() -> dict[str, object]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name in _SHADOWED or name.startswith(tuple(f"{p}." for p in _SHADOWED))
    }


@pytest.fixture
def space_app() -> Iterator[object]:
    """A freshly imported ``hf_space.app`` module, with no lifespan run.

    Function-scoped on purpose: the tests write the module globals that
    ``lifespan`` would otherwise set, and a shared module would carry one
    test's stubs into the next.

    Teardown restores *only* the two shadowed package trees. An earlier version
    dropped every module that appeared during the import, which swept up the
    sklearn and scipy submodules the Space happened to load first and left
    those packages half-initialised for the next test — a fixture doing more
    damage than the thing it was isolating.
    """
    saved_shadowed = _shadowed_modules()
    saved_path = list(sys.path)

    _purge_shadowed()
    sys.path.insert(0, str(SPACE_ROOT))
    try:
        spec = importlib.util.spec_from_file_location(
            _APP_MODULE, SPACE_ROOT / "app.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[_APP_MODULE] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = saved_path
        sys.modules.pop(_APP_MODULE, None)
        _purge_shadowed()
        sys.modules.update(saved_shadowed)
