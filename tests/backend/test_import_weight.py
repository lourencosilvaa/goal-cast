"""Guards that the backend stays free of ML libraries at import time.

The Render image installs only the ``backend`` dependency group — no pandas,
numpy, scikit-learn, scipy or xgboost.  If any backend module imports one of
those at module scope, the container cannot boot on the slim environment.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Libraries deliberately excluded from the `backend` dependency group.
ML_MODULES = ["pandas", "numpy", "sklearn", "scipy", "xgboost"]

_PROBE = """
import sys
import src.backend.main  # noqa: F401
loaded = [name for name in {modules!r} if name in sys.modules]
print(",".join(loaded))
"""


def _ml_modules_loaded_by_importing_backend() -> list[str]:
    """Import the FastAPI app in a clean interpreter, report ML libs pulled in."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(modules=ML_MODULES)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Importing src.backend.main failed:\n{result.stderr}"
    )
    output = result.stdout.strip()
    return output.split(",") if output else []


class TestBackendImportWeight:

    def test_importing_app_pulls_in_no_ml_libraries(self):
        assert _ml_modules_loaded_by_importing_backend() == []

    def test_inference_module_has_no_module_level_pandas_import(self):
        source = (
            PROJECT_ROOT / "src" / "backend" / "api" / "inference.py"
        ).read_text()
        module_level_imports = [
            line
            for line in source.splitlines()
            if line.startswith(("import ", "from ")) and "pandas" in line
        ]
        assert module_level_imports == []
