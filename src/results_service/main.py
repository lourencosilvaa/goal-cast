"""Process entry point — ``uvicorn src.results_service.main:app``.

Building the app at import time is deliberate. If the configuration is wrong
or ``RESULTS_SERVICE_API_KEY`` is unset, the process exits during start-up
instead of serving errors, which is the difference between a failed deploy
that is visible in the platform's logs and a silent one that answers every
request with nothing.

The factory itself lives in :mod:`src.results_service.app` so that importing
it — in a test, or to build a second app with a different runtime — does not
require a real environment.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# The service is started as a module path, so the project root has to be
# importable before the first `src.` import resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.results_service.app import create_app  # noqa: E402
from src.results_service.factory import load_runtime  # noqa: E402

app = create_app(load_runtime())
